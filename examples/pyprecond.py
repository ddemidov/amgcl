#!/usr/bin/env python
import sys, argparse

import numpy   as np
import pyamgcl as amg
from scipy.io import mmread, mmwrite
from scipy.sparse.linalg import lgmres

#----------------------------------------------------------------------------
parser = argparse.ArgumentParser(sys.argv[0])

parser.add_argument('-A,--matrix', dest='A', required=True, help='System matrix in MatrixMarket format')
parser.add_argument('-f,--rhs',    dest='f', required=True, help='RHS in MatrixMarket format')
parser.add_argument('-o,--out',    dest='x', help='Output file name')
parser.add_argument('-p,--prm',    dest='p', help='AMGCL parameters: key1=val1 key2=val2', nargs='+', default=[])

args = parser.parse_args(sys.argv[1:])

#----------------------------------------------------------------------------
# Load problem
A = mmread(args.A)
f = mmread(args.f).flatten()

# Parse parameters
prm = {p[0]: p[1] for p in map(lambda s: s.split('='), args.p)}

# Create preconditioner
P = amg.make_preconditioner(A, prm)
print(P)

cbinfo = dict(iters=0, error=0)
def cb(xi):
    cbinfo['iters'] += 1
    cbinfo['error'] = np.linalg.norm(f - A * xi) / np.linalg.norm(f)
    print('{0}: {1}'.format(cbinfo['iters'], cbinfo['error']))

x,info = lgmres(A, f, M=P, maxiter=100, tol=1e-8, callback=cb)

# Save the solution
if args.x: mmwrite(args.x, x.reshape((-1,1)))
