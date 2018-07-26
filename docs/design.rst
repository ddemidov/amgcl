Design Principles
=================

AMGCL uses the compile-time `policy-based design`_ approach, which allows users
of the library to compose their own version of the AMG method from the provided
components. This also allows for easily extending the library when required.

Components
----------

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

To illustrate this, here is an example of defining a solver type that
combines a BiCGStab iterative method [Barr94]_ preconditioned with smoothed
aggregation AMG that uses SPAI(0) (sparse approximate inverse smoother)
[BrGr02]_ as relaxation. The solver uses the
:cpp:class:`amgcl::backend::builtin` backend (accelerated with OpenMP), and
double precision scalars as value type.

.. code-block:: cpp

    #include <amgcl/backend/builtin.hpp>
    #include <amgcl/make_solver.hpp>
    #include <amgcl/amg.hpp>
    #include <amgcl/coarsening/smoothed_aggregation.hpp>
    #include <amgcl/relaxation/spai0.hpp>
    #include <amgcl/solver/bicgstab.hpp>

    typedef amgcl::backend::builtin<double> Backend;

    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<Backend>
        > Solver;
        
.. _`policy-based design`: https://en.wikipedia.org/wiki/Policy-based_design
.. _OpenMP: https://www.openmp.org/
.. _OpenCL: https://www.khronos.org/opencl/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit
.. _CRS: http://netlib.org/linalg/html_templates/node91.html

Parameters
----------

Each component in AMGCL defines its own parameters by declaring a ``param``
subtype. When a class wraps several subclasses, it includes parameters of its
children into its own ``param``. For example, parameters for the
:cpp:class:`amgcl::make_solver\<Precond, Solver>` are declared as

.. code-block:: cpp

    struct params {
        typename Precond::params precond;
        typename Solver::params solver;
    };

Knowing that, you can easily lookup parameter definitions and set parameters
for individual components. For example, we can set the desired tolerance and
maximum number of iterations for the iterative solver in the above example like
this:

.. code-block:: cpp

    Solver::params prm;
    prm.solver.tol = 1e-3;
    prm.solver.maxiter = 10;
    Solver solve( std::tie(n, ptr, col, val), prm );
