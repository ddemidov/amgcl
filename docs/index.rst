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

Referencing
-----------

.. [Demi19] Demidov, Denis. `AMGCL: An efficient, flexible, and extensible algebraic multigrid implementation <https://doi.org/10.1134/S1995080219050056>`_. Lobachevskii Journal of Mathematics 40.5 (2019): 535-546. `pdf <https://rdcu.be/bHFsY>`_, :download:`bib <demidov19.bib>`
.. [Demi20] Demidov, Denis. `AMGCL – A C++ library for efficient solution of large sparse linear systems <https://doi.org/10.1016/j.simpa.2020.100037>`_. Software Impacts, 6:100037, November 2020. :download:`bib <demidov20.bib>`
.. [DeMW20] Demidov, Denis, Lin Mu, and Bin Wang. `Accelerating linear solvers for Stokes problems with C++ metaprogramming <https://doi.org/10.1016/j.jocs.2020.101285>`_. Journal of Computational Science (2020): 101285. :download:`bib <demidov-mu-wang-20.bib>`

Contents:
---------

.. toctree::
    :maxdepth: 2

    amg_overview
    design
    components
    tutorial
    examples
    benchmarks
    bibliography
