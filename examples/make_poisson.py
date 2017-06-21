#!/usr/bin/python

import numpy as np
from scipy.sparse import csr_matrix

def numba_jit_if_available():
    try:
        from numba import jit
        return jit
    except ImportError:
        return lambda f: f

#----------------------------------------------------------------------------
# Assemble matrix for Poisson problem in a unit square
#----------------------------------------------------------------------------
@numba_jit_if_available()
def make_poisson_2d(n=64):
    nnz = 5 * n**2 - 4 * n

    ptr = np.zeros(n**2+1, dtype=np.int32)
    col = np.zeros(nnz,    dtype=np.int32)
    val = np.zeros(nnz,    dtype=np.float64)
    rhs = np.ones (n**2,   dtype=np.float64)

    idx  = 0
    head = 0

    for j in range(0, n):
        for i in range(0, n):
            if j > 0:
                col[head] = idx - n
                val[head] = -1.0/4.0
                head += 1

            if i > 0:
                col[head] = idx - 1
                val[head] = -1.0/4.0
                head += 1

            col[head] = idx
            val[head] = 1.0
            head += 1

            if i + 1 < n:
                col[head] = idx + 1
                val[head] = -1.0/4.0
                head += 1

            if j + 1 < n:
                col[head] = idx + n
                val[head] = -1.0/4.0
                head += 1

            idx += 1
            ptr[idx] = head

    return ( csr_matrix( (val, col, ptr) ), rhs )

#----------------------------------------------------------------------------
# Assemble matrix for Poisson problem in a unit cube
#----------------------------------------------------------------------------
@numba_jit_if_available()
def make_poisson_3d(n=64):
    nnz = 7 * n**3 - 6 * n**2

    ptr = np.zeros(n**3+1, dtype=np.int32)
    col = np.zeros(nnz,    dtype=np.int32)
    val = np.zeros(nnz,    dtype=np.float64)
    rhs = np.ones (n**3,   dtype=np.float64)

    idx  = 0
    head = 0

    for k in range(0, n):
        for j in range(0, n):
            for i in range(0, n):
                if k > 0:
                    col[head] = idx - n**2
                    val[head] = -1.0/6.0
                    head += 1

                if j > 0:
                    col[head] = idx - n
                    val[head] = -1.0/6.0
                    head += 1

                if i > 0:
                    col[head] = idx - 1
                    val[head] = -1.0/6.0
                    head += 1

                col[head] = idx
                val[head] = 1.0
                head += 1

                if i + 1 < n:
                    col[head] = idx + 1
                    val[head] = -1.0/6.0
                    head += 1

                if j + 1 < n:
                    col[head] = idx + n
                    val[head] = -1.0/6.0
                    head += 1

                if k + 1 < n:
                    col[head] = idx + n**2
                    val[head] = -1.0/6.0
                    head += 1

                idx += 1
                ptr[idx] = head

    return ( csr_matrix( (val, col, ptr) ), rhs )

if __name__ == "__main__":
    import sys
    import argparse
    from scipy.io import mmwrite

    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('-n,--size',   dest='n', default='32',  help='Size of problem to generate')
    parser.add_argument('-A,--matrix', dest='A', default='A',   help='Output matrix filename')
    parser.add_argument('-b,--rhs',    dest='b', default='b',   help='Output rhs filename')
    parser.add_argument('-d,--dim',    dest='d', default='3',   help='Problem dimension (2 or 3)')

    args = parser.parse_args(sys.argv[1:])

    if args.d == 2:
        (A, b) = make_poisson_2d(int(args.n))
    else:
        (A, b) = make_poisson_3d(int(args.n))

    mmwrite(args.A, A)
    mmwrite(args.b, b.reshape((A.shape[0],1)))

