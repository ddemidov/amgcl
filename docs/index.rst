AMGCL documentation
===================

AMGCL is a header-only C++ library for solving large sparse linear systems with
algebraic multigrid (AMG) method. AMG is one of the most effective iterative
methods for solution of equation systems arising, for example, from
discretizing PDEs on unstructured grids [Stue99]_, [TrOS01]_. The method can be
used as a black-box solver for various computational problems, since it does
not require any information about the underlying geometry. AMG is often used
not as a standalone solver but as a preconditioner within an iterative solver
(e.g.  Conjugate Gradients, BiCGStab, or GMRES).

AMGCL builds the AMG hierarchy on a CPU and then transfers it to one of the
provided backends. This allows for transparent acceleration of the solution
phase with help of OpenCL, CUDA, or OpenMP technologies. Users may provide
their own backends which enables tight integration between AMGCL and the user
code.

The library source code is available under MIT license at
https://github.com/ddemidov/amgcl.

Contents:
---------

.. toctree::
    :maxdepth: 2

    tutorial
    components
    runtime
    deflation
    examples
    bibliography
    indices
