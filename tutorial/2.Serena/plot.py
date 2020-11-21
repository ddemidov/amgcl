#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('Serena.mtx')

fig, (ax1, ax2) = subplots(2, 1, sharex=True, figsize=(8,10), gridspec_kw=dict(height_ratios=[4,1]))
ax1.spy(A, marker='.', markersize=0.25, alpha=0.2)
axins = ax1.inset_axes([0.55, 0.55, 0.4, 0.4])
axins.spy(A, marker='o', markersize=3, alpha=0.5)
n = (A.shape[0] // 3 // 3) * 3
axins.set_xlim([n - 0.5, n + 44.5])
axins.set_ylim([n - 0.5, n + 44.5])
axins.invert_yaxis()
axins.set_xticklabels('')
axins.set_yticklabels('')
ax1.indicate_inset_zoom(axins)

ax2.semilogy(A.diagonal())
ax2.set_ylabel('Diagonal')

tight_layout()
savefig('Serena.png')
