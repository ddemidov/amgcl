Design Principles
=================

A lot of linear solver software packages are either developed in C or Fortran,
or provide C-compatible application programming interface (API). The low-level
API is stable and compatible with most of the programming languages. However,
this also has some disadvantages: the fixed interfaces usually only support the
predefined set of cases that the developers have thought of in advance. For an
example, BLAS specification has separate sets of functions that deal with
single, double, complex, or double complex precision values, but it is
impossible to work with mixed precision inputs or with user-defined or
third-party custom types. Another common drawback of large scientific packages
is that users have to adopt the datatypes provided by the framework in order to
work with it, which steepens the learning curve and introduces additional
integration costs, such as the necessity to copy the data between various
formats.

AMGCL is using modern C++ programming techniques in order to create flexible
and efficient API. The users may easily extend the library or use it with their
own datatypes. The following design pronciples are used throughout the code:

- *Policy-based design* [Alex00]_ of public library classes such as
  ``amgcl::make_solver`` or ``amgcl::amg`` allows the library users to compose
  their own customized version of the iterative solver and preconditioner from
  the provided components and easily extend and customize the library by
  providing their own implementation of the algorithms.
- Preference for *free functions* as opposed to member functions [Meye05]_,
  combined with *partial template specialization* allows to extend the library
  operations onto user-defined datatypes and to introduce new algorithmic
  components when required.
- The *backend* system of the library allows expressing the algorithms such as
  Krylov iterative solvers or multigrid relaxation methods in terms of generic
  parallel primitives which facilitates transparent acceleration of the
  solution phase with OpenMP, OpenCL, or CUDA technologies.
- One level below the backends are *value types*: AMGCL supports systems with
  scalar, complex, or block value types both in single and double precision.
  Arithmetic operations necessary for the library work may also be extended
  onto the user-defined types using template specialization.


Policy-based design
-------------------

.. code-block:: cpp
    :name: lst_composition
    :caption: Policy-based design illustration: creating customized solvers from AMGCL components
    :linenos:

    // CG solver preconditioned with ILU0
    typedef amgcl::make_solver<
        amgcl::relaxation::as_preconditioner<
            amgcl::backend::builtin<double>,
            amgcl::relaxation::ilu0
            >,
        amgcl::solver::cg<
            amgcl::backend::builtin<double>
            >
        > Solver1;

    // GMRES solver preconditioned with AMG
    typedef amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::gmres<
            amgcl::backend::builtin<double>
            >
        > Solver2;

Available solvers and preconditioners in AMGCL are composed by the library user
from the provided components. For example, the most frequently used class
template ``amgcl::make_solver<P,S>`` binds together an iterative solver ``S``
and a preconditioner ``P`` chosen by the user. To illustrate this,
:numref:`lst_composition` defines a conjugate gradient iterative solver
preconditioned with an incomplete LU decomposition with zero fill-in in lines 2
to 10. The builtin backend (parallelized with OpenMP) with double precision is
used both for the solver and the preconditioner. This approach allows the user
not only to select any of the preconditioners/solvers provided by AMGCL, but
also to use their own custom components, as long they conform to the generic
AMGCL interface. In paticular, the preconditioner class has to provide a
constructor that takes the system matrix, the preconditioner parameters
(defined as a subtype of the class, see below), and the backend parameters. The
iterative solver constructor should take the size of the system matrix, the
solver parameters, and the backend parameters.

This approach is used not only at the user-facing level of the library, but in
any place where using interchangeable components makes sense.
Lines 13 to 22 in
:numref:`lst_composition` show the declaration of GMRES iterative solver
preconditioned with the algebraic multigrid (AMG). Smoothed aggregation is used
as the AMG coarsening strategy, and diagonal sparse approximate inverse is used
on each level of the multigrid hierarchy as a smoother. Similar to the solver
and the preconditioner, the AMG components (coarsening and relaxation) are
specified as template parameters and may be customized by the user.

.. code-block:: cpp
    :caption: Example of parameter declaration in AMGCL components
    :name: lst_declparams

    template <class P, class S>
    struct make_solver {
        struct params {
            typename P::params precond;
            typename S::params solver;
        };
    };

Besides compile-time composition of the AMGCL algorithms described above, the
library user may need to specify runtime parameters for the constructed
algorithms.  This is done with the ``params`` structure declared by each of the
components as its subtype. Each parameter usually has a reasonable default
value. When a class is composed from several components, it includes the
parameters of its dependencies into its own ``params`` struct.  This allows to
provide a unified interface to the parameters of various AMGCL algorithms.
:numref:`lst_declparams` shows how the parameters are declared for the
``amgcl::make_solver<P,S>`` class. :numref:`lst_params` shows an example of how
the parameters for the preconditioned GMRES solver from
:numref:`lst_composition` may be specified.  Namely, the number of the GMRES
iterations before restart is set to 50, the relative residual threshold is set
to :math:`10^{-6}`, and the strong connectivity threshold
:math:`\varepsilon_{str}` for the smoothed aggregation is set to
:math:`10^{-3}`.  The rest of the parameters are left with their default
values.

.. code-block:: cpp
    :caption: Setting parameters for AMGCL components
    :name: lst_params

    // Set the solver parameters
    Solver2::params prm;
    prm.solver.M = 50;
    prm.solver.tol = 1e-6;
    prm.precond.coarsening.aggr.eps_strong = 1e-3;

    // Instantiate the solver
    Solver2 S(A, prm);

Free functions and partial template specialization
--------------------------------------------------

Using free functions as opposed to class methods allows to decouple the library
functionality from specific classes and enables support for third-party
datatypes within the library [Meye05]_. Moving the implementation from the free
function into a struct template specialization provides more control over the
mapping between the input datatype and the specific specific version of the
algorithm.  For example, constructors of AMGCL classes may accept an arbitrary
datatype as input matrix, as long as the implementations of several basic
functions supporting the datatype have been provided. Some of the free
functions that need to be implemented are ``amgcl::backend::rows(A)``,
``amgcl::backend::cols(A)`` (returning the number of rows and columns for the
matrix), or ``amgcl::backend::row_begin(A,i)`` (returning iterator over the
nonzero values for the matrix row). :numref:`lst_crs_adapter` shows an
implementation of ``amgcl::backend::rows()`` function for the case when the
input matrix is specified as a ``std::tuple(n,ptr,col,val)`` of matrix size
``n``, pointer vector ``ptr`` containing row offsets into the column index and
value vectors, and the column index and values vectors ``col`` and ``val`` for
the nonzero matrix entries.  AMGCL provides adapters for several common input
matrix formats, such as ``Eigen::SparseMatrix`` from Eigen_,
``Epetra_CrsMatrix`` from Trilinos_ Epetra, and it is easy to adapt a
user-defined datatype.

.. _Eigen: http://eigen.tuxfamily.org/
.. _Trilinos: https://trilinos.github.io/

.. code-block:: cpp
    :caption: Implementation of ``amgcl::backend::rows()`` free function for the CRS tuple
    :name: lst_crs_adapter

    // Generic implementation of the rows() function.
    // Works as long as the matrix type provides rows() member function.
    template <class Matrix, class Enable = void>
    struct rows_impl {
        static size_t get(const Matrix &A) {
            return A.rows();
        }
    };

    // Returns the number of rows in a matrix.
    template <class Matrix>
    size_t rows(const Matrix &matrix) {
        return rows_impl<Matrix>::get(matrix);
    }

    // Specialization of rows_impl template for a CRS tuple.
    template < typename N, typename PRng, typename CRng, typename VRng >
    struct rows_impl< std::tuple<N, PRng, CRng, VRng> >
    {
        static size_t get(const std::tuple<N, PRng, CRng, VRng> &A) {
            return std::get<0>(A);
        }
    };

Backends
--------

A backend in AMGCL is a class that binds datatypes like matrix and vector with
parallel primitives like matrix-vector product, linear combination of vectors,
or inner product computation. The backend system is implemented using the free
functions combined with template specialization approach from the previous
section, which decouples implementation of common parallel primitives from
specific datatypes used in the supported backends. This allows to adopt
third-party or user-defined datatypes for use within AMGCL without any
modification.  For example, in order to switch to the CUDA backend in
\cref{lst:composition}, we just need to replace
``amgcl::backend::builtin<double>`` with ``amgcl::backend::cuda<double>``.

Algorithm setup in AMGCL is performed using internal data structures. As soon
as the setup is completed, the necessary objects (mostly matrices and vectors)
are transferred to the backend datatypes. Solution phase of the algorithms is
expressed in terms of the predefined parallel primitives which makes it
possible to switch parallelization technology (such as OpenMP, CUDA, or OpenCL)
simply by changing the backend template parameter of the algorithm. For
example, the residual norm :math:`\epsilon = ||f - Ax||` in AMGCL is computed
using ``amgcl::backend::residual()`` and ``amgcl::backend::inner_product()``
primitives:

.. code-block:: cpp

    backend::residual(f, A, x, r);
    auto e = sqrt(backend::inner_product(r, r));

Value types
-----------

Value type concept allows to generalize AMGCL algorithms onto complex or
non-scalar systems. A value type defines a number of overloads for common math
operations, and is used as a template parameter for a backend. Most often, a
value type is simply a builtin ``double`` or ``float`` atomic value, but
it is also possible to use small statically sized matrices when the system
matrix has a block structure, which may decrease the setup time and the overall
memory footprint, increase cache locality, or improve convergence
ratio.

Value types are used during both the setup and the solution phases. Common
value type operations are defined in ``amgcl::math`` namespace, similar to how
backend operations are defined in ``amgcl::backend``. Examples of such
operations are ``amgcl::math::norm()`` or ``amgcl::math::adjoint()``.
Arithmetic operations like multiplication or addition are defined as operator
overloads.  AMGCL algorithms at the lowest level are expressed in terms of the
value type interface, which makes it possible to switch precision of the
algorithms, or move to complex values, simply by adjusting template parameter
of the selected backend.

The generic implementation of the value type operations also makes it possible
to use efficient third party implementations of the block value arithmetics.
For example, using statically sized Eigen_ matrices instead of builtin
``amgcl::static_matrix`` as block value type may improve performance in case
of relatively large blocks, since the Eigen_ library supports SIMD
vectorization.

Runtime interface
-----------------

The compile-time configuration of AMGCL solvers is not always convenient,
especially if the solvers are used inside a software package or another
library. The runtime interface allows to shift some of the configuraton
decisions to runtime. The classes inside :cpp:any:`amgcl::runtime` namespace
correspond to their compile-time alternatives, but the only template parameter
you need to specify is the backend.

Since there is no way to know the parameter structure at compile time, the
runtime classes accept parameters only in form of
``boost::property_tree::ptree``. The actual components of the method are set
through the parameter tree as well. For example, the solver above could be
constructed at runtime in the following way:

.. code-block:: cpp

    #include <amgcl/backend/builtin.hpp>
    #include <amgcl/make_solver.hpp>
    #include <amgcl/amg.hpp>
    #include <amgcl/coarsening/runtime.hpp>
    #include <amgcl/relaxation/runtime.hpp>
    #include <amgcl/solver/runtime.hpp>

    typedef amgcl::backend::builtin<double> Backend;

    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::runtime::coarsening::wrapper,
            amgcl::runtime::relaxation::wrapper
            >,
        amgcl::runtime::solver::wrapper<Backend>
        > Solver;

    boost::property_tree::ptree prm;

    prm.put("solver.type", "bicgstab");
    prm.put("solver.tol", 1e-3);
    prm.put("solver.maxiter", 10);
    prm.put("precond.coarsening.type", "smoothed_aggregation");
    prm.put("precond.relax.type", "spai0");

    Solver solve( std::tie(n, ptr, col, val), prm );

