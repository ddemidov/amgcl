Algebraic Multigrid
===================

Here we outline the basic principles behind the Algebraic Multigrid (AMG)
method [BrMH85]_, [Stue99]_.  Consider a system of linear algebraic equations
in the form

.. math::

    Au = f

where :math:`A` is an :math:`n \times n` matrix. Multigrid methods are based on
the recursive use of two-grid scheme, which combines

* *Relaxation*, or *smoothing iteration*: a simple iterative method such as
  Jacobi or Gauss-Seidel; and
* *Coarse grid correction*: solving residual equation on a coarser grid.
  Transfer between grids is described with *transfer operators* :math:`P`
  (*prolongation* or *interpolation*) and :math:`R` (*restriction*).

A setup phase of a generic algebraic multigrid (AMG) algorithm may be described
as follows:

* Start with a system matrix :math:`A_1 = A`.
* While the matrix :math:`A_i` is too big to be solved directly:
    1. Introduce prolongation operator :math:`P_i`, and restriction operator
       :math:`R_i`.
    2. Construct coarse system using Galerkin operator:
       :math:`A_{i+1} = R_i A_i P_i`.
* Construct a direct solver for the coarsest system :math:`A_L`.

Note that in order to construct the next level in the AMG hierarchy, we only
need to define transfer operators :math:`P` and :math:`R`. Also, the
restriction operator is often chosen to be a transpose of the prolongation
operator: :math:`R=P^T`.

Having constructed the AMG hierarchy, we can use it to solve the system as
follows:

* Start at the finest level with initial approximation :math:`u_1 = u^0`.
* Iterate until convergence (*V-cycle*):
    * At each level of the grid hiearchy, finest-to-coarsest:
          1. Apply a couple of smoothing iterations (*pre-relaxation*) to the
             current solution :math:`u_i = S_i(A_i, f_i, u_i)`.
          2. Find residual :math:`e_i = f_i - A_i u_i` and restrict it to the
             RHS on the coarser level: :math:`f_{i+1} = R_i e_i`.
    * Solve the corasest system directly: :math:`u_L = A_L^{-1} f_L`.
    * At each level of the grid hiearchy, coarsest-to-finest:
          1. Update the current solution with the interpolated solution from the
             coarser level: :math:`u_i = u_i + P_i u_{i+1}`.
          2. Apply a couple of smoothing iterations (*post-relaxation*) to the
             updated solution: :math:`u_i = S_i(A_i, f_i, u_i)`.

More often AMG is not used standalone, but as a preconditioner with an
iterative Krylov subspace method. In this case single V-cycle is used as a
preconditioning step.

So, in order to fully define an AMG method, we need to choose transfer
operators :math:`P` and :math:`R`, and smoother :math:`S`.
