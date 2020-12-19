Using near null-space vectors (MPI version)
-------------------------------------------

Let us look at how to use the near null-space vectors in the MPI version of the
solver for the elasticity problem (see :doc:`Nullspace`). The following points
need to be kept in mind:

- The near null-space vectors need to be partitioned (and reordered) similar to
  the RHS vector.
- Since we are using coordinates of the discretization grid nodes for the
  computation of the rigid body modes, in order to be able to do this locally
  we need to partition the system in such a way that DOFs from a single grid
  node are owned by the same MPI process. In this case this means we need to do
  a block-wise partitioning with a :math:`3\times3` blocks.
- It is more convenient to partition the coordinate matrix and then to compute
  the rigid body modes.

The listing below shows the complete source code for the MPI elasticity solver
(`tutorial/5.Nullspace/nullspace_mpi.cpp`_)

.. _tutorial/5.Nullspace/nullspace_mpi.cpp: https://github.com/ddemidov/amgcl/blob/master/tutorial/5.Nullspace/nullspace_mpi.cpp
.. _ParMETIS: http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview
.. _PT-SCOTCH: https://www.labri.fr/perso/pelegrin/scotch/

.. literalinclude:: ../../tutorial/5.Nullspace/nullspace_mpi.cpp
   :caption: The MPI solution of the elasticity problem
   :language: cpp
   :linenos:

In lines 44--49 we split the system into approximately equal chunks of rows,
while making sure the chunk sizes are divisible by 3 (the number of DOFs per
grid node). This is a naive paritioning that will be improved a bit later:

We read the parts of the system matrix, the RHS vector, and the grid node
coordinates that belong to the current MPI process in lines 52--61. The
backends for the iterative solver and the preconditioner and the solver type
are declared in lines 72--82.  In lines 85--86 we create the distributed
version of the matrix from the local CRS arrays. After that, we are ready to
partition the system using AMGCL wrapper for either ParMETIS_ or PT-SCOTCH_
libraries (lines 91--123).  Note that we are reordering the coordinate matrix
``coo`` in the same way the RHS vector is reordered, even though the coordinate
matrix has three times less rows than the system matrix. We can do this because
the coordinate matrix is stored in the row-major order, and each row of the
matrix has three coordinates, which means the total number of elements in the
matrix is equal to the number of elements in the RHS vector, and we can apply
our block-wise partitioning to the coordinate matrix.

The coordinates for the current MPI domain are converted into the rigid body
modes in lines 135--136, after which we are ready to setup the solver (line
140) and solve the system (line 152). Below is the output of the compiled
program::

    $ export OMP_NUM_THREADS=1
    $ mpirun -np 4 nullspace_mpi A.bin b.bin C.bin 
    Matrix A.bin: 81657x81657
    RHS b.bin: 81657x1
    Coords C.bin: 27219x3
    Partitioning[ParMETIS] 4 -> 4
    Type:             CG
    Unknowns:         19965
    Memory footprint: 311.95 K

    Number of levels:    3
    Operator complexity: 1.53
    Grid complexity:     1.10

    level     unknowns       nonzeros
    ---------------------------------
        0        81657        3171111 (65.31%) [4]
        1         7824        1674144 (34.48%) [4]
        2          144          10224 ( 0.21%) [4]

    Iters: 104
    Error: 9.26388e-09

    [Nullspace:       2.833 s] (100.00%)
    [ self:           0.070 s] (  2.48%)
    [  partition:     0.230 s] (  8.10%)
    [  read:          0.009 s] (  0.32%)
    [  setup:         1.081 s] ( 38.15%)
    [  solve:         1.443 s] ( 50.94%)
