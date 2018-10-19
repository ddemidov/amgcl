#!/usr/bin/env python
import sys, argparse
import numpy   as np
import pyamgcl as amg
from time import time
from scipy.io import mmread, mmwrite
from scipy.sparse.linalg import lgmres
from make_poisson import *

class timeit:
    profile = {}
    def __init__(self, desc):
        self.desc = desc
        self.tic  = time()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        toc = time()
        timeit.profile[self.desc] = timeit.profile.get(self.desc, 0.0) + (toc - self.tic)

    @staticmethod
    def report():
        print('\n---------------------------------')
        total = sum(timeit.profile.values())
        for k,v in sorted(timeit.profile.items()):
            print('{0:>22}: {1:>8.3f}s ({2:>5.2f}%)'.format(k, v, 100 * v / total))
        print('---------------------------------')
        print('{0:>22}: {1:>8.3f}s'.format('Total', total))

#----------------------------------------------------------------------------
parser = argparse.ArgumentParser(sys.argv[0])

parser.add_argument('-A,--matrix', dest='A', help='System matrix in MatrixMarket format')
parser.add_argument('-f,--rhs',    dest='f', help='RHS in MatrixMarket format')
parser.add_argument('-n,--size',   dest='n', type=int, default=64, help='The size of the Poisson problem to solve when no system matrix is given')
parser.add_argument('-o,--out',    dest='x', help='Output file name')
parser.add_argument('-p,--prm',    dest='p', help='AMGCL parameters: key1=val1 key2=val2', nargs='+', default=[])

args = parser.parse_args(sys.argv[1:])

#----------------------------------------------------------------------------
if args.A:
    with timeit('Read problem'):
        A = mmread(args.A)
        f = mmread(args.f).flatten() if args.f else np.ones(A.shape[0])
else:
    with timeit('Generate problem'):
        A,f = make_poisson_3d(args.n)

# Parse parameters
prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.p)}

# Create preconditioner
with timeit('Setup solver'):
    P = amg.amgcl(A, prm)
print(P)

iters = [0]
def count_iters(x):
    iters[0] += 1

# Solve the system for the RHS
with timeit('Solve the problem'):
    x,info = lgmres(A, f, M=P, maxiter=100, tol=1e-8, atol=1e-8, callback=count_iters)

print('{0}: {1:.6e}'.format(iters[0], np.linalg.norm(f - A * x) / np.linalg.norm(f)))

# Save the solution
if args.x:
    with timeit('Save the result'):
        mmwrite(args.x, x.reshape((-1,1)))

timeit.report()
