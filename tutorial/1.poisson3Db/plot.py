#!/usr/bin/env python
from pylab import *
from scipy.io import mmread

A = mmread('poisson3Db.mtx')
figure(figsize=(8,8))
spy(A, marker='.', markersize=0.25, alpha=0.2)
tight_layout()
savefig('Poisson3D.png')
