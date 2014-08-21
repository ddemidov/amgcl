#!/usr/bin/python

from sys import argv
import numpy   as np
import pyamgcl as amg
from scipy.sparse import *
from pylab import *

# Assemble problem
if len(argv[1:]) > 0:
    n = int(argv[1])
else:
    n = 256

n2 = n * n

A = dok_matrix((n2, n2), dtype = np.float64)

boundaries = [0, n-1]

idx = 0
for i in xrange(0, n):
    for j in xrange(0, n):
        if i in boundaries or j in boundaries:
            A[idx,idx] = 1
        else:
            A[idx,idx-n] = -1
            A[idx,idx-1] = -1
            A[idx,idx  ] =  4
            A[idx,idx+1] = -1
            A[idx,idx+n] = -1

        idx += 1

A = A.tocsr()

# Setup preconditioner
P = amg.precond(
        amg.backend.builtin,
        amg.coarsening.smoothed_aggregation,
        amg.relaxation.spai0,
        amg.params(), A.indptr.astype(np.int), A.indices.astype(np.int), A.data
        )

# Setup solver
S = amg.solver(
        amg.backend.builtin,
        amg.solver_type.bicgstab,
        amg.params(), n2
        )

# Solve
rhs = np.ones([n2])
rhs[0   ] = 0
rhs[n2-1] = 0

x = np.zeros([n2])

S.solve(P, rhs, x)

# Plot result
figure(num=1, figsize=(7,7))
imshow(x.reshape((n,n)), origin='lower')
colorbar()

show()
