#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('CoupCons3D.mtx')
figure(figsize=(8,8))
spy(A, marker='.', markersize=0.25, alpha=0.2)
ax = gca()

zn = 60
x1 = 147071.5
y1 = 63711.5
x2 = 333407.5
y2 = 333407.5

ax1 = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
ax1.spy(A, marker='o', markersize=3, alpha=0.5)
ax1.set_xlim([x1, x1 + zn])
ax1.set_ylim([y1, y1 + zn])
ax1.invert_yaxis()
ax1.set_xticklabels('')
ax1.set_yticklabels('')

ax2 = ax.inset_axes([0.05, 0.05, 0.4, 0.4])
ax2.spy(A, marker='o', markersize=3, alpha=0.5)
ax2.set_xlim([x2, x2 + zn])
ax2.set_ylim([y2, y2 + zn])
ax2.invert_yaxis()
ax2.set_xticklabels('')
ax2.set_yticklabels('')

ax.indicate_inset_zoom(ax1)
ax.indicate_inset_zoom(ax2)

tight_layout()
savefig('CoupCons3D.png')
