#!/usr/bin/env python

from pylab import *
from scipy.sparse import csr_matrix, lil_matrix

n = 4
m = n//2
h = 1/(n-1)

ptr = [0]
col = []
val = []
mpi = []

k = 0
for j in range(n):
    for i in range(n):
        if j > 0:
            col.append(k-n)
            val.append(-1/h**2)
        if i > 0:
            col.append(k-1)
            val.append(-1/h**2)
        col.append(k)
        val.append(4/h**2)
        if i < n-1:
            col.append(k+1)
            val.append(-1/h**2)
        if j < n-1:
            col.append(k+n)
            val.append(-1/h**2)


        k += 1
        ptr.append(len(col))
        mpi.append((j>=m) * 2 + (i>=m))

A = csr_matrix((val, col, ptr))

fig, ((ax1, ax2), (ax3, ax4)) = subplots(2,2, figsize=(8,8))
C = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.invert_yaxis()

for v in (0.25, 0.50, 0.75):
    ax1.axhline(v)
    ax1.axvline(v)
for i in range(4):
    ax1.axhspan(i*0.25, (i+1)*0.25, color=C[i], lw=0, alpha=0.25)

k = 0
for j in range(4):
    for i in range(4):
        ax1.text(0.05 + i * 0.25, 0.05 + j * 0.25, f'{k}', ha='center', va='center')
        k += 1

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.invert_yaxis()

for v in (0.25, 0.50, 0.75):
    ax2.axhline(v)
    ax2.axvline(v)
k = 0
for j in range(2):
    for i in range(2):
        ax2.axhspan(i*0.5, (i+1)*0.5, j * 0.5, (j+1)*0.5, color=C[k], lw=0, alpha=0.25)
        k += 1

idx = {j:i for i,j in enumerate(sorted(range(16), key=lambda i: mpi[i]))}

k = 0
for j in range(4):
    for i in range(4):
        ax2.text(0.05 + i * 0.25, 0.05 + j * 0.25, f'{k}', ha='center', va='center')
        ax2.text(0.2 + i * 0.25, 0.2 + j * 0.25, f'{idx[k]}', ha='center', va='center')
        k += 1

ax3.spy(A, marker='o', markersize=5)
for i in range(4):
    ax3.axhspan(i * 4 - 0.5, (i + 1) * 4 - 0.5, color=C[i], lw=0, alpha=0.25, zorder=0)
for i in range(1,4):
    ax3.axhline(i * 4 - 0.5)
    ax3.axvline(i * 4 - 0.5, 1 - 0.25 * (i - 1), 1 - 0.25 * (i + 1))
ax3.xaxis.tick_bottom()

ax1.set_title('Naive domain partitioning')
ax2.set_title('Optimal domain partitioning')
ax3.set_title('Naive matrix partitioning')
ax4.set_title('Optimal matrix partitioning')

I = lil_matrix((16,16))
for i,j in idx.items():
    I[i,j] = 1
ax4.spy(I * A * I.T, marker='o', markersize=5)
for i in range(4):
    ax4.axhspan(i * 4 - 0.5, (i + 1) * 4 - 0.5, color=C[i], lw=0, alpha=0.25, zorder=0)
for i in range(1,4):
    ax4.axhline(i * 4 - 0.5)
    ax4.axvline(i * 4 - 0.5, 1 - 0.25 * (i - 1), 1 - 0.25 * (i + 1))
ax4.xaxis.tick_bottom()

tight_layout()
show()
