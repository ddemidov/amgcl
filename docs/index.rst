AMGCL documentation
===================

AMGCL is a header-only C++ library for solving large sparse linear systems with
algebraic multigrid (AMG) method. AMG is one of the most effective iterative
methods for solution of equation systems arising, for example, from
discretizing PDEs on unstructured grids [BrMH85]_, [Stue99]_, [TrOS01]_. The
method can be used as a black-box solver for various computational problems,
since it does not require any information about the underlying geometry. AMG is
often used not as a standalone solver but as a preconditioner within an
iterative solver (e.g.  Conjugate Gradients, BiCGStab, or GMRES).

The library has minimal dependencies, and provides both shared-memory and
distributed memory (MPI) versions of the algorithms.  The AMG hierarchy is
constructed on a CPU and then is transferred into one of the provided backends.
This allows for transparent acceleration of the solution phase with help of
OpenCL, CUDA, or OpenMP technologies. Users may provide their own backends
which enables tight integration between AMGCL and the user code.

The source code is available under liberal MIT license at
https://github.com/ddemidov/amgcl.

Contents:
---------

.. toctree::
    :maxdepth: 2

    amg_overview
    design
    bibliography
