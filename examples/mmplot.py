#!/usr/bin/env python
import argparse
from pathlib import Path
from pylab import *
from scipy.io import mmread
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Plot a MatrixMarket file')
parser.add_argument('matrix', help='input matrix file')
parser.add_argument('-b,--binary', dest='binary', action='store_true', help='input matrix is binary')
parser.add_argument('-z,--zoom', dest='zoom', nargs=2, metavar=('START', 'SIZE'), type=int, help='zoom position (start and size on the diagonal)')
parser.add_argument('-o,--output', dest='output', help='output image file')
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

if args.zoom:
    axz = ax1.inset_axes([0.55, 0.55, 0.4, 0.4])
    axz.spy(A, marker='o', markersize=1, alpha=0.5)
    n,m = args.zoom
    axz.set_xlim([n - 0.5, n + m - 0.5])
    axz.set_ylim([n - 0.5, n + m - 0.5])
    axz.invert_yaxis()
    axz.set_xticklabels('')
    axz.set_yticklabels('')
    ax1.indicate_inset_zoom(axz)

ax2.semilogy(abs(A.diagonal()))
ax2.set_ylabel('Diagonal')
tight_layout()

if args.output:
    savefig(args.output)
else:
    show()
