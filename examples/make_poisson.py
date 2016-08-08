#!/usr/bin/python

import numpy as np
from scipy.sparse import csr_matrix

#----------------------------------------------------------------------------
# Assemble matrix for Poisson problem in a unit cube
#----------------------------------------------------------------------------
def make_poisson(n=64):
    nnz = 7 * n**3 - 6 * n**2

    ptr = np.zeros(n**3+1, dtype=np.int32)
    col = np.zeros(nnz,    dtype=np.int32)
    val = np.zeros(nnz,    dtype=np.float64)
    rhs = np.ones (n**3,   dtype=np.float64)

    bnd = (0, n-1)

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

    parser.add_argument('-n,--size',   dest='n', default='256', help='Size of problem to generate')
    parser.add_argument('-A,--matrix', dest='A', default='A',   help='Output matrix filename')
    parser.add_argument('-b,--rhs',    dest='b', default='b',   help='Output rhs filename')

    args = parser.parse_args(sys.argv[1:])

    (A, b) = make_poisson(int(args.n))

    mmwrite(args.A, A)
    mmwrite(args.b, b.reshape((A.shape[0],1)))

