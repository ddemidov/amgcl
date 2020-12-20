Iterative Solvers
=================

An iterative solver is a Krylov subspace method that may be combined with a
:doc:`preconditioner <preconditioners>` in order to solve the linear system.

All iterative solvers in AMGCL have two template parameters, ``Backend`` and
``InnerProduct``.  The ``Backend`` template parameter specifies the
:doc:`backend <backends>` to target, and the ``InnerProduct`` parameter is used
to select the implementation of the inner product to use with the solver. The
correct implementation should be automatically selected by the library
depending on whether the solver is used in a shared or distributed memory
setting.

All solvers provide similar interface described below:

.. cpp:function:: constructor( \
                  size_t n, \
                  const params &prm = params(), \
                  const backend_params &bprm = backend_params())

   The solver constructor. Takes the size of the system to solve, the solver
   parameters and the backend parameters.

.. cpp:function:: template <class Matrix, class Precond, class VectorRHS, class VectorX> \
                  std::tuple<size_t, scalar_type> operator()( \
                      const Matrix &A, \
                      const Precond &P, \
                      const VectorRHS &rhs, \
                      const VectorX &x)

   Computes the solution for the given system matrix ``A`` and the right-hand
   side ``rhs``.

   Returns the number of iterations made and the achieved
   relative residual as a ``std::tuple<size_t,scalar_type>``. The solution
   vector ``x`` provides initial approximation on input and holds the computed
   solution on output.

.. cpp:function:: template <class Precond, class VectorRHS, class VectorX> \
                  std::tuple<size_t, scalar_type> operator()( \
                      const Precond &P, \
                      const VectorRHS &rhs, \
                      const VectorX &x)

   Computes the solution for the given right-hand side ``rhs``. The matrix that
   was used to create the preconditioner ``P`` is used as the system
   matrix.

   Returns the number of iterations made and the achieved
   relative residual as a ``std::tuple<size_t,scalar_type>``. The solution
   vector ``x`` provides initial approximation on input and holds the computed
   solution on output.


AMGCL implementats the following iterative solvers:

CG
--

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::cg

   .. rubric:: Include ``<amgcl/solver/cg.hpp>``

   The Conjugate Gradient method is an effective method for symmetric positive
   definite systems. It is probably the oldest and best known of the
   nonstationary methods [Barr94]_, [Saad03]_.

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

BiCGStab
--------

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::bicgstab

   .. rubric:: Include ``<amgcl/solver/bicgstab.hpp>``

   The BiConjugate Gradient Stabilized method (BiCGStab) was developed to solve
   nonsymmetric linear systems while avoiding the often irregular convergence
   patterns of the Conjugate Gradient Squared method [Barr94]_.

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: bool check_after = false

         Always do at least one iteration

      .. cpp:member:: amgcl::preconditioner::side::type pside = amgcl::preconditioner::side::right

         Preconditioner kind (left/right)

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

BiCGStab(L)
-----------

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::bicgstabl

   .. rubric:: Include ``<amgcl/solver/bicgstabl.hpp>``

   This is a generalization of the BiCGStab method [SlDi93]_, [Fokk96]_. For
   :math:`L=1`, this algorithm coincides with BiCGStab. In some situations it
   may be profitable to take :math:`L>2`. Although the steps of BiCGStab(L) are
   more expensive for larger L, numerical experiments indicate that, in certain
   situations, due to a faster convergence, for instance, BiCGStab(4) performs
   better than BiCGStab(2).

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: int L = 2

         The order of the method

      .. cpp:member:: scalar_type delta = 0

         Threshold used to decide when to refresh computed residuals.

      .. cpp:member:: bool convex = true

         Use a convex function of the MinRes and OR polynomials after the BiCG
         step instead of default MinRes

      .. cpp:member:: amgcl::preconditioner::side::type pside = amgcl::preconditioner::side::right

         Preconditioner kind (left/right)

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

GMRES
-----

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::gmres

   .. rubric:: Include ``<amgcl/solver/gmres.hpp>``

   The Generalized Minimal Residual method is an extension of MINRES (which is
   only applicable to symmetric systems) to unsymmetric systems. Like MINRES,
   it generates a sequence of orthogonal vectors, but in the absence of
   symmetry this can no longer be done with short recurrences; instead, all
   previously computed vectors in the orthogonal sequence have to be retained.
   For this reason, “restarted” versions of the method are used [Barr94]_.

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: int M = 30

         The number of iterations before restart

      .. cpp:member:: amgcl::preconditioner::side::type pside = amgcl::preconditioner::side::right

         Preconditioner kind (left/right)

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

"Loose" GMRES (LGMRES)
----------------------

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::lgmres

   .. rubric:: Include ``<amgcl/solver/lgmres.hpp>``

   The residual vectors at the end of each restart cycle of restarted GMRES
   often alternate direction in a cyclic fashion, thereby slowing convergence.
   LGMRES is an implementation of a technique for accelerating the convergence
   of restarted GMRES by disrupting this alternating pattern. The new algorithm
   resembles a full conjugate gradient method with polynomial preconditioning,
   and its implementation requires minimal changes to the standard restarted
   GMRES algorithm [BaJM05]_.

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: unsigned K = 3

         Number of vectors to carry between inner GMRES iterations.  According
         to [BaJM05]_, good values are in the range of 1-3.  However, if you
         want to use the additional vectors to accelerate solving multiple
         similar problems, larger values may be beneficial.

      .. cpp:member:: bool always_reset = true

         Reset augmented vectors between solves.  If the solver is used to
         repeatedly solve similar problems, then keeping the augmented vectors
         between solves may speed up subsequent solves.  This flag, when set,
         resets the augmented vectors at the beginning of each solve.

      .. cpp:member:: int M = 30

         The number of iterations before restart

      .. cpp:member:: amgcl::preconditioner::side::type pside = amgcl::preconditioner::side::right

         Preconditioner kind (left/right)

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

Flexible GMRES (FGMRES)
-----------------------

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::fgmres

   .. rubric:: Include ``<amgcl/solver/fgmres.hpp>``

   Often, the application of the preconditioner P is a result of some
   unspecified computation, possibly another iterative process. In such cases,
   it may well happen that P is not a constant operator. The preconditioned
   iterative solvers may not converge if P is not constant. There are a number
   of variants of iterative procedures developed in the literature that can
   accommodate variations in the preconditioner, i.e., that allow the
   preconditioner to vary from step to step. Such iterative procedures are
   called “flexible” iterations. The method implements flexible variant of the
   GMRES algorithm [Saad03]_.

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: int M = 30

         The number of iterations before restart

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

IDR(s)
------

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::idrs

   .. rubric:: Include ``<amgcl/solver/idrs.hpp>``

   This is a very stable and efficient IDR(s) variant as described in [GiSo11]_.
   The Induced Dimension Reduction method, IDR(s), is a robust and efficient
   short-recurrence Krylov subspace method for solving large nonsymmetric
   systems of linear equations.

   IDR(s) compared to  BI-CGSTAB/BiCGStab():

   - Faster.
   - More robust.
   - More flexible. 

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: unsigned s = 4

         Dimension of the shadow space in IDR(s).

      .. cpp:member:: scalar_type omega = 0.7

         Computation of omega.

         - If omega = 0, a standard minimum residual step is performed.
         - If omega > 0, omega is increased if the cosine of the angle between Ar and r < omega.

      .. cpp:member:: bool smoothing = false

         Specifies if residual smoothing must be applied.

      .. cpp:member:: bool replacement = false

         Residual replacement.  Determines the residual replacement strategy.
         If true, the recursively computed residual is replaced by the true
         residual.

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

Richardson iteration
--------------------

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::richardson

   .. rubric:: Include ``<amgcl/solver/richardson.hpp>``

   The preconditioned Richardson iterative method

   .. math:: x^{i+1} = x^{i} + \omega P(f - A x^{i})

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.

      .. cpp:member:: scalar_type damping = 1.0

         The damping factor :math:`\omega`

      .. cpp:member:: size_t maxiter = 100

         The maximum number of iterations

      .. cpp:member:: scalar_type tol = 1e-8

         Target relative residual error :math:`\varepsilon = \frac{||f - Ax||}{|| f ||}`

      .. cpp:member:: scalar_type abstol = std::numeric_limits<scalar_type>::min()

         Target absolute residual error :math:`\varepsilon = ||f - Ax||`

      .. cpp:member:: bool ns_search = false

         Ignore the trivial solution ``x=0`` when the RHS is zero.
         Useful when searching for the null-space vectors of the system.

      .. cpp:member:: bool verbose = false

         Output the current iteration number and relative residual during
         solution.

PreOnly
-------

.. cpp:class:: template <class Backend, class InnerProduct = amgcl::detail::default_inner_product> \
               amgcl::solver::preonly

   .. rubric:: Include ``<amgcl/solver/preonly.hpp>``

   Only apply the preconditioner once. This is not very useful as a standalone
   solver, but may be used in composite preconditioners as a nested solver, so
   that the composite preconditioner itself remains linear and may be used with
   a non-flexible iterative solver.

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The solver parameters.
