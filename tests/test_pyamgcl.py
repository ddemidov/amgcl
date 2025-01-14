#!/usr/bin/python3

import unittest
import numpy   as np
import pyamgcl as amg
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import bicgstab, LinearOperator

def make_problem(n):
    h   = 1.0 / (n - 1)
    n2  = n * n
    nnz = n2 + 4 * (n - 2) * (n - 2)

    ptr = np.zeros(n2 + 1, dtype=np.int32)
    col = np.zeros(nnz,    dtype=np.int32)
    val = np.zeros(nnz,    dtype=np.float64)
    rhs = np.ones (n2,     dtype=np.float64)

    bnd = (0, n-1)

    col_stencil = np.array([-n, -1, 0,  1,  n])
    val_stencil = np.array([-1, -1, 4, -1, -1]) / (h * h)

    idx  = 0
    head = 0

    for i in range(0, n):
        for j in range(0, n):
            if i in bnd or j in bnd:
                col[head] = idx
                val[head] = 1
                rhs[idx]  = 0
                head += 1
            else:
                col[head:head+5] = col_stencil + idx
                val[head:head+5] = val_stencil
                head += 5

            idx += 1
            ptr[idx] = head

    return ( csr_matrix( (val, col, ptr) ), rhs )

class TestPyAMGCL(unittest.TestCase):
    def test_solver(self):
        A, rhs = make_problem(100)

        for stype in ('bicgstab', 'lgmres'):
            for rtype in ('spai0', 'ilu0'):
                P = amg.amgcl(A, prm={'relax.type': rtype})
                # Setup solver
                solve = amg.solver(P, prm=dict(type=stype, tol=1e-3, maxiter=1000))

                # Solve
                x = solve(rhs)

    def test_preconditioner(self):
        A, rhs = make_problem(100)

        for rtype in ('spai0', 'ilu0'):
            P = amg.amgcl(A, prm={'relax.type': rtype})
            # Solve
            #
            # The scipy devs changed their API in a non-backwards-compatible way: "tol" was removed in favor of "rtol":
            #   https://docs.scipy.org/doc/scipy-1.15.0/reference/generated/scipy.sparse.linalg.bicgstab.html
            #
            # The docs aren't clear about how to preserve the old behavior. I
            # try the old method, and if that fails, I try the new method
            try:
                x,info = bicgstab(A, rhs, M=P, tol=1e-3)
            except TypeError:
                x,info = bicgstab(A, rhs, M=P, rtol=1e-3)
            self.assertTrue(info == 0)

            # Check residual
            self.assertTrue(norm(rhs - A * x) / norm(rhs) < 1e-3)

if __name__ == "__main__":
    unittest.main()
