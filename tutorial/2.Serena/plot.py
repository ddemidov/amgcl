#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('Serena.mtx')
figure(figsize=(8,8))
spy(A, marker='.', markersize=0.25, alpha=0.2)
ax = gca()
axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
axins.spy(A, marker='o', markersize=3, alpha=0.5)
n = (A.shape[0] // 3 // 3) * 3
axins.set_xlim([n - 0.5, n + 44.5])
axins.set_ylim([n - 0.5, n + 44.5])
axins.invert_yaxis()
axins.set_xticklabels('')
axins.set_yticklabels('')
ax.indicate_inset_zoom(axins)
tight_layout()
savefig('Serena.png')
