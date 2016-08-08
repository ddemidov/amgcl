#!/usr/bin/env python
import sys, argparse

import numpy   as np
import pyamgcl as amg
from scipy.io import mmread, mmwrite
from make_poisson import *

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
    A = mmread(args.A)
    f = mmread(args.f).flatten() if args.f else ones(A.rows())
else:
    A,f = make_poisson(args.n)

# Parse parameters
prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.p)}

# Create solver
solve = amg.make_solver(A, prm)
print(solve)

# Solve the system for the RHS
x = solve(f)

error = np.linalg.norm(f - A * x) / np.linalg.norm(f)
print("{0.iters}: {0.error:.6e} / {1:.6e}".format(solve, error))

# Save the solution
if args.x: mmwrite(args.x, x.reshape((-1,1)))
