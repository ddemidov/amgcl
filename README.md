amgcl
=====

Simple AMG hierarchy builder. May be used as a standalone solver or as a
preconditioner. Currently the only supported iterative solvers are the ones
from ViennaCL (http://viennacl.sourceforge.net).

Eigen (http://eigen.tuxfamily.org) or VexCL (https://github.com/ddemidov/vexcl)
matrix/vector containers are supported with ViennaCL solvers. See
examples/eigen.cpp and examples/vexcl.cpp for respective examples.
