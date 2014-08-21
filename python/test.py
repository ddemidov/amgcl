#!/usr/bin/python

from sys import argv
import numpy   as np
import pyamgcl as amg
from scipy.sparse import *
from pylab import *
from time import time

# Assemble problem
if len(argv[1:]) > 0:
    n = int(argv[1])
else:
    n = 256

tic = time()
n2 = n * n
nnz = n2 + 4 * (n - 2) * (n - 2)

ptr = np.zeros(n2 + 1).astype(np.int32)
col = np.zeros(nnz).astype(np.int32)
val = np.zeros(nnz)

boundaries = (0, n-1)
col_stencil = np.array([-n, -1, 0,  1,  n])
val_stencil = np.array([-1, -1, 4, -1, -1])

idx  = 0
head = 0
for i in xrange(0, n):
    for j in xrange(0, n):
        if i in boundaries or j in boundaries:
            col[head] = idx
            val[head] = 1
            head += 1
        else:
            col[head:head+5] = idx + col_stencil
            val[head:head+5] = val_stencil
            head += 5

        idx += 1
        ptr[idx] = head

print "Assemble: %.2f" % (time() - tic)

# Setup preconditioner
tic = time()
P = amg.precond(
        amg.backend.builtin,
        amg.coarsening.smoothed_aggregation,
        amg.relaxation.spai0,
        amg.params(), ptr, col, val
        )

# Setup solver
S = amg.solver(
        amg.backend.builtin,
        amg.solver_type.bicgstab,
        amg.params(), n2
        )
print "Setup: %.2f" % (time() - tic)

# Solve
rhs = np.ones([n2])
rhs[0   ] = 0
rhs[n2-1] = 0

x = np.zeros([n2])

tic = time()
S.solve(P, rhs, x)
print "Solve: %.2f" % (time() - tic)

# Plot result
figure(num=1, figsize=(7,7))
imshow(x.reshape((n,n)), origin='lower')
colorbar()

show()
