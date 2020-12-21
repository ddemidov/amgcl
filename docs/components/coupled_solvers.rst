Coupling Solvers with Preconditioners
=====================================

These classes provide a convenient way to couple an iterative solver and a
preconditioner. This may be used both for convenience and as a building block
for a composite :doc:`preconditioner <preconditioners>`.

make_solver
-----------

.. cpp:class:: template <class Precond, class IterSolver> \
               amgcl::make_solver

   .. rubric:: Include ``<amgcl/make_solver.hpp>``

   The class has two template parameters: ``Precond`` and ``IterSolver``, which
   specify the preconditioner and the iterative solver to use. During
   construction of the class, instances of both components are constructed and
   are ready to use as a whole.

   .. cpp:type:: typename Backend::params backend_params

      The backend parameters

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The coupled solver parameters

      .. cpp:member:: typename Precond::params precond

         The preconditioner parameters

      .. cpp:member:: IterSolver::params solver

         The iterative solver parameters

   .. cpp:function:: template <class Matrix> \
                     make_solver(const Matrix &A, \
                                 const params &prm = params(), \
                                 const backend_params &bprm = backend_params())

      The constructor

   .. cpp:function:: template <class Matrix, class VectorRHS, class VectorX> \
                     std::tuple<size_t, scalar_type> operator()( \
                         const Matrix &A, const VectorRHS &rhs, VectorX &x) const

      Computes the solution for the given system matrix ``A`` and the
      right-hand side ``rhs``.  Returns the number of iterations made and
      the achieved residual as a ``std::tuple``. The solution vector
      ``x`` provides initial approximation on input and holds the computed
      solution on output.
      
      The system matrix may differ from the matrix used during
      initialization. This may be used for the solution of non-stationary
      problems with slowly changing coefficients. There is a strong chance
      that a preconditioner built for a time step will act as a reasonably
      good preconditioner for several subsequent time steps [DeSh12]_.

   .. cpp:function:: template <class VectorRHS, class VectorX> \
                     std::tuple<size_t, scalar_type> operator()( \
                         const VectorRHS &rhs, VectorX &x) const

      Computes the solution for the given right-hand side ``rhs``.
      Returns the number of iterations made and the achieved residual as a
      ``std::tuple``. The solution vector ``x`` provides initial
      approximation on input and holds the computed solution on output.

   .. cpp:function:: const Precond& precond() const

      Returns reference to the constructed preconditioner

   .. cpp:function:: const IterSolver& solver() const

      Returns reference to the constructed iterative solver

make_block_solver
-----------------

.. cpp:class:: template <class Precond, class IterSolver> \
               amgcl::make_block_solver

   .. rubric:: Include ``<amgcl/make_block_solver.hpp>``

   Creates coupled solver which targets a block valued backend, but may be
   initialized with a scalar system matrix, and used with scalar vectors.

   The scalar system matrix is transparently converted to the block-valued on
   using the :cpp:func:`amgcl::adapter::block_matrix` adapter in the class
   constructor, and the scalar vectors are reinterpreted to the block-valued
   ones upon application.

   This class may be used as a building block in a composite preconditioner,
   when one (or more) of the subsystems has block values, but has to be
   computed as a scalar matrix.

   The interface is the same as that of :cpp:class:`amgcl::make_solver`.

deflated_solver
---------------

.. cpp:class:: template <class Precond, class IterSolver> \
               amgcl::deflated_solver

   .. rubric:: Include ``<amgcl/deflated_solver.hpp>``

   Creates preconditioned deflated solver. Deflated Krylov subspace methods are
   supposed to solve problems with large jumps in the coefficients on layered
   domains. It appears that the convergence of a deflated solver is independent
   of the size of the jump in the coefficients. The specific variant of the
   deflation method used here is A-DEF2 from [TNVE09]_.

   .. cpp:type:: typename Backend::params backend_params

      The backend parameters

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:class:: params

      The deflated solver parameters

      .. cpp:member:: int nvec = 0

         The number of deflation vectors

      .. cpp:member:: scalar_type *vec = nullptr

         .. close the*

         The deflation vectors stored as a [nvec x n] matrix in row-major order

      .. cpp:member:: typename Precond::params precond

         The preconditioner parameters

      .. cpp:member:: IterSolver::params solver

         The iterative solver parameters

