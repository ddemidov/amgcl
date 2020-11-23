#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('poisson3Db.mtx')

M = 4
n = A.shape[0]
chunk = n // M
domain = [min(n, i*chunk) for i in range(M+1)]

figure(figsize=(8,8))
spy(A, marker='.', markersize=0.25, alpha=0.2)
for i,c in enumerate(('tab:blue', 'tab:orange', 'tab:green', 'tab:red')):
    axhspan(domain[i], domain[i+1], color=c, lw=0, alpha=0.25, zorder=0)

tight_layout()

savefig('Poisson3D_mpi.png')
