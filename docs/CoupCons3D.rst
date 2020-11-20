3D fully coupled poroelastic problem
====================================

This system may be downloaded from the CoupCons3D_ page (use the `Matrix
Market`_ download option). According to the description, the system has been
obtained through a Finite Element transient simulation of a fully coupled
consolidation problem on a three-dimensional domain using Finite Differences
for the discretization in time.  More details available in [FePG09]_ and
[FeJP12]_. The RHS vector for the CoupCons3D problem is not provided, and we
use the RHS vector filled with ones.

The system matrix is non-symmetric and has 416,800 rows and 17,277,420 nonzero
values, which corresponds to an average of 41 nonzeros per row. The matrix
portrait is shown on the figure below.

.. figure:: ../tutorial/3.CoupCons3D/CoupCons3D.png
   :width: 80%
   :name: coupcons3d_spy

   CoupCons3D matrix portrait

.. _CoupCons3D: https://sparse.tamu.edu/Janna/CoupCons3D
.. _Matrix Market: https://math.nist.gov/MatrixMarket
.. _examples/mm2bin: https://github.com/ddemidov/amgcl/blob/master/examples/mm2bin.cpp
.. _examples/solver: https://github.com/ddemidov/amgcl/blob/master/examples/solver.cpp
.. _examples/schur_pressure_correction: https://github.com/ddemidov/amgcl/blob/master/examples/schur_pressure_correction.cpp

.. [FePG09] M. Ferronato, G. Pini, and G. Gambolati. The role of preconditioning in the solution to FE coupled consolidation equations by Krylov subspace methods. International Journal for Numerical and Analytical Methods in Geomechanics 33 (2009), pp. 405-423.
.. [FeJP12] M. Ferronato, C. Janna, and G. Pini. Parallel solution to ill-conditioned FE geomechanical problems. International Journal for Numerical and Analytical Methods in Geomechanics 36 (2012), pp. 422-437.

Once again, lets start our experiments with the `examples/solver`_ utility
after converting the matrix into binary format with `examples/mm2bin`_. The
default options do not seem to work for this problem; switching to LGMRES
solver and increasing the iteration limit does not help either::

    $ solver -B -A CoupCons3D.bin solver.type=lgmres solver.maxiter=1000
    Solver
    ======
    Type:             LGMRES(30,3)
    Unknowns:         416800
    Memory footprint: 120.86 M

    Preconditioner
    ==============
    Number of levels:    4
    Operator complexity: 1.11
    Grid complexity:     1.09
    Memory footprint:    447.17 M

    level     unknowns       nonzeros      memory
    ---------------------------------------------
        0       416800       22322336    404.08 M (90.13%)
        1        32140        2214998     38.49 M ( 8.94%)
        2         3762         206242      3.58 M ( 0.83%)
        3          522          22424      1.03 M ( 0.09%)

    Iterations: 1000
    Error:      0.000960454

    [Profile:      97.684 s] (100.00%)
    [  reading:     0.187 s] (  0.19%)
    [  setup:       0.555 s] (  0.57%)
    [  solve:      96.940 s] ( 99.24%)

What works is using the higher quality relaxation (incomplete LU decomposition
with zero fill-in)::

    $ solver -B -A CoupCons3D.bin solver.type=lgmres solver.maxiter=1000 \
          precond.relax.type=ilu0
    Solver
    ======
    Type:             LGMRES(30,3)
    Unknowns:         416800
    Memory footprint: 120.86 M

    Preconditioner
    ==============
    Number of levels:    4
    Operator complexity: 1.11
    Grid complexity:     1.09
    Memory footprint:    832.12 M

    level     unknowns       nonzeros      memory
    ---------------------------------------------
        0       416800       22322336    751.33 M (90.13%)
        1        32140        2214998     72.91 M ( 8.94%)
        2         3762         206242      6.85 M ( 0.83%)
        3          522          22424      1.03 M ( 0.09%)

    Iterations: 29
    Error:      3.97161e-09

    [Profile:       6.050 s] (100.00%)
    [  reading:     0.187 s] (  3.10%)
    [  setup:       1.715 s] ( 28.34%)
    [  solve:       4.146 s] ( 68.53%)

From the :ref:`coupcons3d_spy` it is obvious that the system matrix has block
structure, with two diagonal subblocks. The upper left subblock contains
333,440 unknowns and seems to have a block structure of its own with small
:math:`4\times4` blocks, and the lower right subblock is a simple diagonal
matrix. From [FePG09]_ it becomes clear that the unknowns from the upper left
subblock correspond to the soil displacement, and the unknowns from the lower
right subblock correspond to pressure, but we do not actually need this
information to proceed with our experiments. Knowing that the matrix has two
large sets of unknowns, we can try the Schur pressure correction preconditioner
that creates separate solvers for each of the system subblocks.
`examples/solver`_ does not support this, but we can use
`examples/schur_pressure_correction`_ for our tests. In the Schur pressure
correction preconditioner terms, the upper left block constitutes the "U"
subsystem, and the lower right block is the "P" subsystem. We need to decide
what solvers to use for these subsystems. Lets start with
:cpp:class:`amgcl::solver::preonly` solver, which "solves" the system by
applying the preconditioner exactly once. This does not work well as a
standalone solver, but is useful as a nested solver. Further, knowing that
``amgcl::relaxation::ilu0`` worked well enough for the whole system, and that
the upper left subblock contains the majority of the matrix complexity, lets
use the same relaxation with the "U" system solver. The "P" system is a simple
diagonal, so single-level relaxation used as a preconditioner should work well
enough here::

    $ schur_pressure_correction -B -A CoupCons3D.bin -m '>333440' -p 
          solver.type=lgmres \
          precond.usolver.solver.type=preonly \
          precond.usolver.precond.relax.type=ilu0 \
          precond.psolver.solver.type=preonly \
          precond.psolver.precond.class=relaxation
    Solver
    ======
    Type:             LGMRES(30,3)
    Unknowns:         416800
    Memory footprint: 120.86 M

    Preconditioner
    ==============
    Schur complement (two-stage preconditioner)
      Unknowns: 416800(83360)
      Nonzeros: 22322336
      Memory:  1.10 G

    [ U ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         333440
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Number of levels:    4
    Operator complexity: 1.14
    Grid complexity:     1.11
    Memory footprint:    674.51 M

    level     unknowns       nonzeros      memory
    ---------------------------------------------
        0       333440       17324768    593.73 M (87.64%)
        1        32140        2214998     72.91 M (11.20%)
        2         3762         206242      6.85 M ( 1.04%)
        3          522          22424      1.03 M ( 0.11%)

    [ P ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         83360
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Relaxation as preconditioner
      Unknowns: 83360
      Nonzeros: 2332064
      Memory:   36.86 M


    Iterations: 29
    Error:      3.97161e-09

    [Profile:                8.272 s] (100.00%)
    [  reading:              0.191 s] (  2.31%)
    [  schur_complement:     8.081 s] ( 97.69%)
    [   self:                0.051 s] (  0.62%)
    [    setup:              1.544 s] ( 18.66%)
    [    solve:              6.486 s] ( 78.41%)

This seems to work, but the performance is worse that the monolithic
preconditioner with ``ilu0`` as relaxation. We did not yet use the fact that
the "U" system has block structure with :math:`4\times4` blocks. Lets try using
a block-valued backend for the "U" subsystem with a ``--ub 4`` command line
option::


    $ schur_pressure_correction -B -A CoupCons3D.bin -m '>333440' --ub 4 -p 
          solver.type=lgmres \
          precond.usolver.solver.type=preonly \
          precond.usolver.precond.relax.type=ilu0 \
          precond.psolver.solver.type=preonly \
          precond.psolver.precond.class=relaxation
    Solver
    ======
    Type:             LGMRES(30,3)
    Unknowns:         416800
    Memory footprint: 120.86 M

    Preconditioner
    ==============
    Schur complement (two-stage preconditioner)
      Unknowns: 416800(83360)
      Nonzeros: 22322336
      Memory:  938.92 M

    [ U ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         83360
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Number of levels:    4
    Operator complexity: 1.40
    Grid complexity:     1.19
    Memory footprint:    482.62 M

    level     unknowns       nonzeros      memory
    ---------------------------------------------
        0        83360        1082798    353.76 M (71.35%)
        1        14473         394281    116.89 M (25.98%)
        2         1380          39160     11.59 M ( 2.58%)
        3           64           1368    395.75 K ( 0.09%)


    [ P ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         83360
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Relaxation as preconditioner
      Unknowns: 83360
      Nonzeros: 2332064
      Memory:   36.86 M


    Iterations: 100
    Error:      3.82965

    [Profile:               17.686 s] (100.00%)
    [  reading:              0.188 s] (  1.07%)
    [  schur_complement:    17.497 s] ( 98.93%)
    [   self:                0.043 s] (  0.24%)
    [    setup:              0.584 s] (  3.30%)
    [    solve:             16.870 s] ( 95.39%)

That did not work. At all. From the experience, sometimes it helps to use
non-smoothed aggregation instead of smoothed aggregation with block valued
systems. Lets try that::

    $ schur_pressure_correction -B -A CoupCons3D.bin -m '>333440' --ub 4 -p 
          solver.type=lgmres \
          precond.usolver.solver.type=preonly \
          precond.usolver.precond.coarsening.type=aggregation \
          precond.usolver.precond.relax.type=ilu0 \
          precond.psolver.solver.type=preonly \
          precond.psolver.precond.class=relaxation
    Solver
    ======
    Type:             LGMRES(30,3)
    Unknowns:         416800
    Memory footprint: 120.86 M

    Preconditioner
    ==============
    Schur complement (two-stage preconditioner)
      Unknowns: 416800(83360)
      Nonzeros: 22322336
      Memory:  842.87 M

    [ U ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         83360
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Number of levels:    4
    Operator complexity: 1.21
    Grid complexity:     1.23
    Memory footprint:    386.57 M

    level     unknowns       nonzeros      memory
    ---------------------------------------------
        0        83360        1082798    313.37 M (82.53%)
        1        14473         184035     53.34 M (14.03%)
        2         4105          39433     11.27 M ( 3.01%)
        3          605           5761      8.60 M ( 0.44%)


    [ P ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         83360
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Relaxation as preconditioner
      Unknowns: 83360
      Nonzeros: 2332064
      Memory:   36.86 M


    Iterations: 12
    Error:      4.68258e-09

    [Profile:                2.554 s] (100.00%)
    [  reading:              0.191 s] (  7.48%)
    [  schur_complement:     2.362 s] ( 92.51%)
    [   self:                0.039 s] (  1.55%)
    [    setup:              0.462 s] ( 18.10%)
    [    solve:              1.861 s] ( 72.87%)

Ok, that is much better! Compared to the monolithic ``ilu0`` preconditioner,
the setup time dropped from 1.7 seconds to 0.5 seconds (3x speedup), and the
solution phase is more than twice faster (1.9 seconds vs 4.2 seconds).
Lets see how this approach translates to the code. We will use the mixed
precision approach, and we will also try if the less expensive (memory-wise)
BiCGStab solver is now able to solve the system. The listing below shows the complete solution and is also available
in `tutorial/3.CoupCons3D/coupcons3d.cpp`_.

.. literalinclude:: ../tutorial/3.CoupCons3D/coupcons3d.cpp
   :caption: The source code for the solution of the CoupCons3D problem.
   :language: cpp
   :linenos:

.. _tutorial/3.CoupCons3D/coupcons3d.cpp: https://github.com/ddemidov/amgcl/blob/master/tutorial/3.CoupCons3D/coupcons3d.cpp

As always, we include all the necessary components in lines 4--20. In the first
lines of the ``main`` function we check that we have enough command line
arguments (the matrix file name and the number of "U" unknowns in the system)
and read the system matrix from the `Matrix Market`_ format.  In lines 55--58
we define the double precision backend for the outer iterative solver, and
single precision backends for the "U" and "P" subsystems. The "U" solver
backend uses :math:`4\times4` block values.

Lines 60--79 contain the definition of the solver with the composite Schur
pressure correction preconditioner. Note that the solver for the "U" subsystem
is declared with :cpp:class:`amgcl::make_block_solver` as opposed to
:cpp:class:`amgcl::make_solver`. The block solver transparently converts its
scalar input matrix (the "U" subblock, extracted from the system matrix using
the ``pmask`` parameter during the Schur pressure correction preconditioner
setup) to the block valued format, and uses the ``reinterpret_cast`` trick that
we have seen in the :doc:`Serena` tutorial to create the block-valued views for
the RHS and the solution vectors.

In lines 82--84 we create the solver parameters. The only parameter that we
need to set here is the ``pmask`` vector that should contrain ``true`` for the
unknowns belonging to the "P" subsystem, and ``false`` for the "U" unknowns.


We instantiate the solver and solve the system in lines 88 and 99.
Here is the output from the compiled program::

    $ ./coupcons3d CoupCons3D.mtx 333440
    Matrix CoupCons3D.mtx: 416800x416800
    Solver
    ======
    Type:             BiCGStab
    Unknowns:         416800
    Memory footprint: 22.26 M

    Preconditioner
    ==============
    Schur complement (two-stage preconditioner)
      Unknowns: 416800(83360)
      Nonzeros: 22322336
      Memory:  549.90 M

    [ U ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         83360
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Number of levels:    4
    Operator complexity: 1.21
    Grid complexity:     1.23
    Memory footprint:    206.09 M

    level     unknowns       nonzeros      memory
    ---------------------------------------------
        0        83360        1082798    167.26 M (82.53%)
        1        14473         184035     28.49 M (14.03%)
        2         4105          39433      6.04 M ( 3.01%)
        3          605           5761      4.30 M ( 0.44%)


    [ P ]
    Solver
    ======
    Type:             PreOnly
    Unknowns:         83360
    Memory footprint: 0.00 B

    Preconditioner
    ==============
    Relaxation as preconditioner
      Unknowns: 83360
      Nonzeros: 2332064
      Memory:   27.64 M


    Iters: 7
    Error: 5.0602e-09

    [CoupCons3D:    14.427 s] (100.00%)
    [  read:        13.010 s] ( 90.18%)
    [  setup:        0.336 s] (  2.33%)
    [  solve:        1.079 s] (  7.48%)

As we can see, the mixed precision approach shaves off about 40% from the setup
time, and the solution time is about 80% faster than our best try with the full
precision solver. Overall, this version of the solver is 4 times faster than
the one with the monolithic preconditioner.
