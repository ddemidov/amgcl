import numpy as np
import pyamgcl as amg
from scipy.sparse import csr_matrix
from time import time

#----------------------------------------------------------------------------
# Assemble matrix for Poisson problem in a unit square
#----------------------------------------------------------------------------
def make_poisson(n=256):
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


#----------------------------------------------------------------------------
n = 1024
(A, b) = make_poisson(n)

tic = time()
solve = amg.make_solver(A)
tm_setup = time() - tic
print solve

tic = time()
x = solve(b)
tm_solve = time() - tic

print "Iterations:", solve.iterations()
print "Error:     ", solve.residual()
print "Setup:      %.2f sec" % tm_setup
print "Solve:      %.2f sec" % tm_solve
