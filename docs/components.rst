Components
==========

AMGCL provides the following components:

* **Backends** -- classes that define matrix and vector types and operations
  necessary during the solution phase of the algorithm. When an AMG hierarchy
  is constructed, it is moved to the specified backend. The approach enables
  transparent acceleration of the solution phase with OpenMP_, OpenCL_, or
  CUDA_ technologies, and also makes tight integration with user-defined data
  structures possible.
* **Value types** -- enable transparent solution of complex or non-scalar
  systems. Most often, a value type is simply a ``double``, but it is possible
  to use small statically-sized matrices as value type, which may increase
  cache-locality, or convergence ratio, or both, when the system matrix has a
  block structure. 
* **Matrix adapters** -- allow AMGCL to construct a solver from some common
  matrix formats. Internally, the CRS_ format is used, but it is easy to adapt
  any matrix format that allows row-wise iteration over its non-zero elements.
* **Coarsening strategies** -- various options for creating coarse systems in
  the AMG hierarchy. A coarsening strategy takes the system matrix :math:`A` at
  the current level, and returns prolongation operator :math:`P` and the
  corresponding restriction operator :math:`R`.
* **Relaxation methods** -- or smoothers, that are used on each level of the
  AMG hierarchy during solution phase.
* **Preconditioners** -- aside from the AMG, AMGCL implements preconditioners
  for some common problem types. For example, there is a Schur complement
  pressure correction preconditioner for Navie-Stokes type problems, or CPR
  preconditioner for reservoir simulations. Also, it is possible to use single
  level relaxation method as a preconditioner.
* **Iterative solvers** -- Krylov subspace methods that may be combined with
  the AMG (or other) preconditioners in order to solve the linear system.

