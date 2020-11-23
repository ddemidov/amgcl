#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('Serena.mtx')

B = 3
M = 4
n = A.shape[0]
chunk = n // M
if chunk % B:
    chunk += B - chunk % B
domain = [min(n, i*chunk) for i in range(M+1)]

figure(figsize=(8,8))
spy(A, marker='.', markersize=0.25, alpha=0.2)
for i in range(4):
    axhspan(domain[i], domain[i+1], alpha=0.1 * i, zorder=0)
tight_layout()
savefig('Serena_mpi.png')
