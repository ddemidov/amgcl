Structural problem (MPI version)
--------------------------------

In this section we look at how to use the MPI version of the AMGCL solver with
the Serena_ system. We have already determined in the :doc:`Serena` section
that the system is best solved with the block-valued backend, and needs to be
scaled so that it has the unit diagonal. The MPI solution will be very closer
to the one we have seen in the :doc:`poisson3DbMPI` section. The only
differences are:

.. _Serena: https://sparse.tamu.edu/Janna/Serena

- The system needs to be scaled so that it has the unit diagonal. This is
  complicated by the fact that the matrix product :math:`D^{-1/2} A D^{-1/2}`
  has to done in the distributed memory environment.
- The solution has to use the block-valued backend, and the partitioning needs
  to take this into account. Namely, the partitioning should not split any of
  the :math:`3\times3` blocks between MPI processes.
- Even though the system is symmetric, the convergence of the CG solver in the
  distributed case stalls at the relative error about :math:`10^{-6}`.
  Switching to the BiCGStab solver helps with the convergence.

The next listing is the MPI version of the Serena_ system solver
(`tutorial/2.Serena/serena_mpi.cpp`_).  In the following paragraphs we
highlight the differences between this version and the code in the
:doc:`poisson3DbMPI` and :doc:`Serena` sections.

.. _tutorial/2.Serena/serena_mpi.cpp: https://github.com/ddemidov/amgcl/blob/master/tutorial/2.Serena/serena_mpi.cpp

.. literalinclude:: ../tutorial/2.Serena/serena_mpi.cpp
   :caption: The MPI solution of the Serena problem
   :language: cpp
   :linenos:

We make sure that the paritioning takes the block structure of the matrix into
account by keeping the number of rows in the initial naive partitioning
divisible by 3 (here the constant ``B`` is equal to 3):

.. literalinclude:: ../tutorial/2.Serena/serena_mpi.cpp
   :language: cpp
   :linenos:
   :lines: 46-53
   :lineno-start: 46

We also create all the distributed matrices using the block values, so the
partitioning naturally is block-aware. We are using the mixed precision
approach, so the preconditioner backend is defined with the single precision:

.. literalinclude:: ../tutorial/2.Serena/serena_mpi.cpp
   :language: cpp
   :linenos:
   :lines: 65-79
   :lineno-start: 65

The scaling is done similarly to how we apply the reordering: first, we find
the diagonal of the local diagonal block on each of the MPI processes, and then
we create the distributed diagonal matrix with the inverted square root of the
system matrix diagonal. After that, the scaled matrix :math:`A_s = D^{-1/2} A
D^{-1/2}` is computed using the :cpp:func:`amgcl::mpi::product()` function.
The scaled RHS vector :math:`f_s = D^{-1/2} f` in principle may be found using
the :cpp:func:`amgcl::backend::spmv()` primitive, but, since the RHS vector in
this case is simply filled with ones, the scaled RHS :math:`f_s = D^{-1/2}`.

.. literalinclude:: ../tutorial/2.Serena/serena_mpi.cpp
   :language: cpp
   :linenos:
   :lines: 85-125
   :lineno-start: 85

Here is the output from the compiled program::

    $ export OMP_NUM_THREADS=1
    $ mpirun -np 4 ./serena_mpi Serena.bin 
    World size: 4
    Matrix Serena.bin: 1391349x1391349
    Partitioning[ParMETIS] 4 -> 4
    Type:             BiCGStab
    Unknowns:         118533
    Memory footprint: 18.99 M

    Number of levels:    4
    Operator complexity: 1.27
    Grid complexity:     1.07

    level     unknowns       nonzeros
    ---------------------------------
        0       463783        7170189 (79.04%) [4]
        1        32896        1752778 (19.32%) [4]
        2         1698         144308 ( 1.59%) [4]
        3           95           4833 ( 0.05%) [4]

    Iterations: 80
    Error:      9.34355e-09

    [Serena MPI:     24.840 s] (100.00%)
    [  partition:     1.159 s] (  4.67%)
    [  read:          0.265 s] (  1.07%)
    [  scale:         0.583 s] (  2.35%)
    [  setup:         0.811 s] (  3.26%)
    [  solve:        22.017 s] ( 88.64%)

The version that uses the VexCL backend should be familiar at this point. Below
is the source code (`tutorial/2.Serena/serena_mpi_vexcl.cpp`_) where the
differences with the builtin backend version are highlighted:

.. _tutorial/2.Serena/serena_mpi_vexcl.cpp: https://github.com/ddemidov/amgcl/blob/master/tutorial/2.Serena/serena_mpi_vexcl.cpp

.. literalinclude:: ../tutorial/2.Serena/serena_mpi_vexcl.cpp
   :caption: The MPI solution of the Serena problem using the VexCL backend
   :language: cpp
   :linenos:
   :emphasize-lines: 4-5,40-52,84-85,100-102,144,168-169,184,193-194

Here is the output of the MPI version with the VexCL backend::

    $ export OMP_NUM_THREADS=1
    $ mpirun -np 2 ./serena_mpi_vexcl_cl Serena.bin 
    0: GeForce GTX 960 (NVIDIA CUDA)
    1: GeForce GTX 1050 Ti (NVIDIA CUDA)
    World size: 2
    Matrix Serena.bin: 1391349x1391349
    Partitioning[ParMETIS] 2 -> 2
    Type:             BiCGStab
    Unknowns:         231112
    Memory footprint: 37.03 M

    Number of levels:    4
    Operator complexity: 1.27
    Grid complexity:     1.07

    level     unknowns       nonzeros
    ---------------------------------
        0       463783        7170189 (79.01%) [2]
        1        32887        1754795 (19.34%) [2]
        2         1708         146064 ( 1.61%) [2]
        3           85           4059 ( 0.04%) [2]

    Iterations: 83
    Error:      9.80582e-09

    [Serena MPI(VexCL):    10.943 s] (100.00%)
    [  partition:           1.357 s] ( 12.40%)
    [  read:                0.370 s] (  3.38%)
    [  scale:               0.729 s] (  6.66%)
    [  setup:               1.966 s] ( 17.97%)
    [  solve:               6.512 s] ( 59.51%)
