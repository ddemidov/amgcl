Benchmarks
==========

The performance of the shared memory and the distributed memory versions of
AMGCL algorithms was tested on two example problems in a three dimensional
space.  The source code for the benchmarks is available at
https://github.com/ddemidov/amgcl_benchmarks.

The first example is the classical 3D Poisson problem. Namely, we look for the
solution of the problem

.. math::

    -\Delta u = 1,

in the unit cube :math:`\Omega = [0,1]^3` with homogeneous Dirichlet boundary
conditions. The problem is dicretized with the finite difference method on a
uniform mesh.

The second test problem is an incompressible 3D Navier-Stokes problem
discretized on a non uniform 3D mesh with a finite element method:

.. math::

    \frac{\partial \mathbf u}{\partial t} + \mathbf u \cdot \nabla \mathbf u +
    \nabla p = \mathbf b,

    \nabla \cdot \mathbf u = 0.

The discretization uses an equal-order tetrahedral Finite Elements stabilized
with an ASGS-type (algebraic subgrid-scale) approach. This results in a linear
system of equations with a block structure of the type

.. math::

    \begin{pmatrix}
        \mathbf K & \mathbf G \\
        \mathbf D & \mathbf S
    \end{pmatrix}
    \begin{pmatrix}
        \mathbf u \\ \mathbf p
    \end{pmatrix}
    =
    \begin{pmatrix}
        \mathbf b_u \\ \mathbf b_p
    \end{pmatrix}

where each of the matrix subblocks is a large sparse matrix, and the blocks
:math:`\mathbf G` and :math:`\mathbf D` are non-square.  The overall system
matrix for the problem was assembled in the Kratos_ multi-physics package
developed in CIMNE, Barcelona. 

.. _Kratos: http://www.cimne.com/kratos/
