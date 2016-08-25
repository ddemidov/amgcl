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
parser.add_argument('-p,--precond', dest='p', help='preconditioner parameters: key1=val1 key2=val2', nargs='+', default=[])
parser.add_argument('-s,--solver',  dest='s', help='solver parameters: key1=val1 key2=val2', nargs='+', default=[])
parser.add_argument('-1,--single-level', dest='single', help='Use single level relaxation as preconditioner', action='store_true', default=False)

args = parser.parse_args(sys.argv[1:])

#----------------------------------------------------------------------------
if args.A:
    A = mmread(args.A)
    f = mmread(args.f).flatten() if args.f else ones(A.rows())
else:
    A,f = make_poisson(args.n)

# Parse parameters
p_prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.p)}
s_prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.s)}

# Create solver/preconditioner pair
S = amg.solver(amg.relaxation(A, p_prm) if args.single else amg.amg(A, p_prm), s_prm)
print(S)

# Solve the system for the RHS
x = S(f)

error = np.linalg.norm(f - A * x) / np.linalg.norm(f)
print("{0.iters}: {0.error:.6e} / {1:.6e}".format(S, error))

# Save the solution
if args.x: mmwrite(args.x, x.reshape((-1,1)))
