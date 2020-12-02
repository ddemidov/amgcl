#!/usr/bin/env python
import argparse
from pathlib import Path
from pylab import *
from scipy.io import mmread
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Plot a MatrixMarket file')
parser.add_argument('matrix', help='input matrix file')
parser.add_argument('-b,--binary', dest='binary', action='store_true', help='input matrix is binary')
args = parser.parse_args()

if args.binary:
    with open(args.matrix, 'rb') as f:
        n, = fromfile(f, int64, 1)
        ptr = fromfile(f, int64, n+1)
        col = fromfile(f, int64, ptr[-1])
        val = fromfile(f, float64, ptr[-1])
        A = csr_matrix((val, col, ptr))
else:
    A = mmread(args.matrix)

fig, (ax1, ax2) = subplots(2, 1, sharex=True, figsize=(8,10), gridspec_kw=dict(height_ratios=[4,1]))
ax1.spy(A, marker='.', markersize=1)
ax1.set_title(f'{Path(args.matrix).stem} (${A.shape[0]} \\times {A.shape[1]}$)')
ax2.semilogy(A.diagonal())
ax2.set_ylabel('Diagonal')
tight_layout()

show()
