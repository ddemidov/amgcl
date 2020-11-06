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

D. Demidov. AMGCL: An efficient, flexible, and extensible algebraic multigrid
implementation. Lobachevskii Journal of Mathematics, 40(5):535–546, May 2019.
doi1_ pdf1_

.. _doi1: https://doi.org/10.1134/S1995080219050056
.. _pdf1: https://rdcu.be/bHFsY

.. code:: bibtex

    @Article{Demidov2019,
        author="Demidov, D.",
        title="AMGCL: An Efficient, Flexible, and Extensible Algebraic Multigrid Implementation",
        journal="Lobachevskii Journal of Mathematics",
        year="2019",
        month="May",
        day="01",
        volume="40",
        number="5",
        pages="535--546",
        issn="1818-9962",
        doi="10.1134/S1995080219050056",
        url="https://doi.org/10.1134/S1995080219050056"
    }

D. Demidov. AMGCL -- A C++ library for efficient solution of large sparse linear
systems.  Software Impacts, 6:100037, November 2020.
doi2_

.. _doi2: https://doi.org/10.1016/j.simpa.2020.100037

.. code:: bibtex

    @article{Demidov2020,
        author = "Denis Demidov",
        title = "AMGCL -- A C++ library for efficient solution of large sparse linear systems",
        journal = "Software Impacts",
        volume = "6",
        pages = "100037",
        year = "2020",
        issn = "2665-9638",
        doi = "10.1016/j.simpa.2020.100037",
        url = "https://doi.org/10.1016/j.simpa.2020.100037"
    }

Contents:
---------

.. toctree::
    :maxdepth: 2

    amg_overview
    design
    components
    examples
    mpi
    benchmarks
    bibliography
