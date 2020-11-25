#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('ucube_4_A.mm')

fig, (ax1, ax2) = subplots(2, 1, sharex=True, figsize=(8,10), gridspec_kw=dict(height_ratios=[4,1]))

ax1.spy(A, marker='.', markersize=0.25, alpha=0.2)
ax1.axhline(456191.5, c='k', ls=':', alpha=0.25)
ax1.axvline(456191.5, c='k', ls=':', alpha=0.25)

n1 = 120
n2 = 65
x1 = 153599.5
y1 = 153599.5
x2 = 456158.5
y2 = 456158.5

az1 = ax1.inset_axes([0.45, 0.65, 0.3, 0.3])
az1.spy(A, marker='o', markersize=1, alpha=0.5)
az1.set_xlim([x1, x1 + n1])
az1.set_ylim([y1, y1 + n1])
az1.invert_yaxis()
az1.set_xticklabels('')
az1.set_yticklabels('')

az2 = ax1.inset_axes([0.05, 0.25, 0.3, 0.3])
az2.spy(A, marker='o', markersize=3, alpha=0.5)
az2.axhline(456191.5, c='k', ls=':', alpha=0.25)
az2.axvline(456191.5, c='k', ls=':', alpha=0.25)
az2.set_xlim([x2, x2 + n2])
az2.set_ylim([y2, y2 + n2])
az2.invert_yaxis()
az2.set_xticklabels('')
az2.set_yticklabels('')

ax1.indicate_inset_zoom(az1)
ax1.indicate_inset_zoom(az2)

ax2.semilogy(abs(A.diagonal()), lw=1)
ax2.set_ylabel('Diagonal')
az3 = ax2.inset_axes([0.2, 0.2, 0.3, 0.6])
az3.semilogy(abs(A.diagonal()), lw=1)
az3.set_xlim(50000,50600)
az3.set_xticklabels('')
az3.set_yticklabels('')
ax2.indicate_inset_zoom(az3)

tight_layout()
savefig('ucube_4.png')
