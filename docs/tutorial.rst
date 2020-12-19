Tutorial
========

In this section we demonstrate the solution of some common types of problems.
The first three problems are matrices from the `SuiteSparse Matrix
Collection`_, which is a widely used set of sparse matrix benchmarks.  The
Stokes problem may be downloaded from the `dataset
<https://doi.org/10.5281/zenodo.4134357>`_ accompanying the [DeMW20]_ paper.
The solution timings used in the sections below were obtained on an Intel Core
i5-3570K CPU. The timings for the GPU backends were obtained with the NVIDIA
GeForce GTX 1050 Ti GPU.

.. _SuiteSparse Matrix Collection: https://sparse.tamu.edu/

.. toctree::
    :maxdepth: 1

    tutorial/poisson3Db
    tutorial/poisson3DbMPI
    tutorial/Serena
    tutorial/SerenaMPI
    tutorial/CoupCons3D
    tutorial/Stokes
    tutorial/Nullspace
    tutorial/NullspaceMPI
