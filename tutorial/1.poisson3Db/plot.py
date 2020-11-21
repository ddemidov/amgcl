#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('poisson3Db.mtx')

fig, (ax1, ax2) = subplots(2, 1, sharex=True, figsize=(8,10), gridspec_kw=dict(height_ratios=[4,1]))
ax1.spy(A, marker='.', markersize=0.25, alpha=0.2)
ax2.semilogy(A.diagonal())
ax2.set_ylabel('Diagonal')
tight_layout()

savefig('Poisson3D.png')
