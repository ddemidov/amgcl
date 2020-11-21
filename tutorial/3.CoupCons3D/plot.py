#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('CoupCons3D.mtx')

fig, (ax1, ax2) = subplots(2, 1, sharex=True, figsize=(8,10), gridspec_kw=dict(height_ratios=[4,1]))

ax1.spy(A, marker='.', markersize=0.25, alpha=0.2)

zn = 60
x1 = 147071.5
y1 = 63711.5
x2 = 333407.5
y2 = 333407.5

az1 = ax1.inset_axes([0.55, 0.55, 0.4, 0.4])
az1.spy(A, marker='o', markersize=3, alpha=0.5)
az1.set_xlim([x1, x1 + zn])
az1.set_ylim([y1, y1 + zn])
az1.invert_yaxis()
az1.set_xticklabels('')
az1.set_yticklabels('')

az2 = ax1.inset_axes([0.05, 0.05, 0.4, 0.4])
az2.spy(A, marker='o', markersize=3, alpha=0.5)
az2.set_xlim([x2, x2 + zn])
az2.set_ylim([y2, y2 + zn])
az2.invert_yaxis()
az2.set_xticklabels('')
az2.set_yticklabels('')

ax1.indicate_inset_zoom(az1)
ax1.indicate_inset_zoom(az2)

ax2.semilogy(A.diagonal())
ax2.set_ylabel('Diagonal')

tight_layout()
savefig('CoupCons3D.png')
