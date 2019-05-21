#!/usr/bin/env python
import sys, argparse

import numpy   as np
import pyamgcl as amg
from scipy.io import mmread, mmwrite
from time import time
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
parser.add_argument('-p,--precond', dest='p', help='preconditioner parameters: key1=val1 key2=val2', nargs='+', default=[])
parser.add_argument('-s,--solver',  dest='s', help='solver parameters: key1=val1 key2=val2', nargs='+', default=[])

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
p_prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.p)}
s_prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.s)}

# Create solver/preconditioner pair
with timeit('Setup solver'):
    S = amg.solver(amg.amg(A, p_prm), s_prm)
print(S)

# Solve the system for the RHS
with timeit('Solve the problem'):
    x = S(f)

error = np.linalg.norm(f - A * x) / np.linalg.norm(f)
print("{0.iters}: {0.error:.6e} / {1:.6e}".format(S, error))

# Save the solution
if args.x:
    with timeit('Save the result'):
        mmwrite(args.x, x.reshape((-1,1)))

timeit.report()
