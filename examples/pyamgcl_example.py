#!/usr/bin/python

import numpy as np
import pyamgcl as amg
from scipy.sparse import csr_matrix
from time import time

from make_poisson import make_poisson



n = 1024
(A, b) = make_poisson(n)

tic = time()
solve = amg.make_solver(A)
tm_setup = time() - tic
print solve

tic = time()
x = solve(b)
tm_solve = time() - tic

print "Iterations:", solve.iterations()
print "Error:     ", solve.residual()
print "Setup:      %.2f sec" % tm_setup
print "Solve:      %.2f sec" % tm_solve
