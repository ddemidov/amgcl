#!/usr/bin/python

import numpy as np
import pyamgcl as amg

# Assemble problem
n = 16 * 1024;

ptr = np.zeros([n + 1]).astype(np.int)
col = np.zeros([n*3 - 4]).astype(np.int)
val = np.zeros([n*3 - 4])

j = 0
for i in xrange(0,n):
    if (i == 0 or i == n - 1):
        col[j] = i
        val[j] = 1.0

        j = j + 1
    else:
        col[j+0] = i - 1
        col[j+1] = i
        col[j+2] = i + 1

        val[j+0] = -1.0
        val[j+1] =  2.0
        val[j+2] = -1.0

        j = j + 3

    ptr[i+1] = j

# Setup preconditioner
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
        amg.params(), n
        )

# Solve
rhs = np.ones([n])
rhs[0]   = 0
rhs[n-1] = 0

x = np.zeros([n])

S.solve(P, rhs, x)

print x
