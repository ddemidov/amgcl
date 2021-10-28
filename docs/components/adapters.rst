Matrix Adapters
===============

A matrix adapter allows AMGCL to construct a solver from some common matrix
formats. Internally, the CRS_ format is used, but it is easy to adapt any
matrix format that allows row-wise iteration over its non-zero elements.

.. _CRS: http://netlib.org/linalg/html_templates/node91.html

Tuple of CRS arrays
-------------------

.. rubric:: Include ``<amgcl/adapter/crs_tuple.hpp>``

It is possible to use a ``std::tuple`` of CRS arrays as input matrix for any of
the AMGCL algorithms. The CRS arrays may be stored either as STL containers:

.. code-block:: cpp

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;

    Solver S( std::tie(n, ptr, col, val) );

or as ``amgcl::iterator_range``, which makes it possible to adapt raw pointers:

.. code-block:: cpp

    int    *ptr;
    int    *col;
    double *val;

    Solver S( std::make_tuple(n,
                          amgcl::make_iterator_range(ptr, ptr + n + 1),
                          amgcl::make_iterator_range(col, col + ptr[n]),
                          amgcl::make_iterator_range(val, val + ptr[n])
                          ) );

Zero copy
---------

.. rubric:: Include ``<amgcl/adapter/zero_copy.hpp>``

.. cpp:function:: template <class Ptr, class Col, class Val> \
                  std::shared_ptr< amgcl::backend::crs<Val> > zero_copy(size_t n, const Ptr *ptr, const Col *col, const Val *val)

   .. closing*

   Returns a shared pointer to the sparse matrix in internal AMGCL format. The
   matrix may be directly used for constructing AMGCL algorithms. ``Ptr`` and
   ``Col`` have to be 64bit integral datatypes (signed or unsigned). In case
   the :cpp:class:`amgcl::backend::builtin` backend is used, no data will be
   copied from the CRS arrays, so it is the user's responsibility to make sure
   the pointers are alive until the AMGCL algorithm is destroyed.


Block matrix
------------

.. rubric:: Include ``<amgcl/adapter/block_matrix.hpp>``

.. cpp:function:: template <class BlockType, class Matrix> \
                  block_matrix_adapter<Matrix, BlockType> block_matrix(const Matrix &A)

   Converts scalar-valued matrix to a block-valued one on the fly. The adapter
   allows to iterate the rows of the scalar-valued matrix as if the matrix was
   stored using the block values. The rows of the input matrix have to be
   sorted column-wise.

Scaled system
-------------

.. rubric:: Include ``<amgcl/adapter/scaled_problem.hpp>``

.. cpp:function:: template <class Backend, class Matrix> \
                  auto scaled_diagonal(const Matrix &A, const typename Backend::params &bprm = typename Backend::params())

   Returns a scaler object that may be used to scale the system so that the
   matrix has unit diagonal:

   .. math::

      A_s = D^{1/2} A D^{1/2}

   where :math:`D` is the matrix diagonal. This keeps the matrix symmetrical.
   The RHS also needs to be scaled, and the solution of the system has to be
   postprocessed:

   .. math::

      D^{1/2} A D^{1/2} y = D^{1/2} b, \quad x = D^{1/2} y

   The scaler object may be used to scale both the matrix:

   .. code-block:: cpp

      auto A = std::tie(rows, ptr, col, val);
      auto scale = amgcl::adapter::scale_diagonal<Backend>(A, bprm);
      
      // Setup solver
      Solver solve(scale.matrix(A), prm, bprm);

   and the RHS:

   .. code-block:: cpp

      // option 1: rhs is untouched
      solve(*scale.rhs(b), x);
      
      // option 2: rhs is prescaled in-place
      scale(b);
      solve(b, x);

   .. close*

   The solution vector has to be postprocessed afterwards:

   .. code-block:: cpp

      // postprocess the solution in-place:
      scale(x);

Reordered system
----------------

.. rubric:: Include ``<amgcl/adapter/reorder.hpp>``

.. cpp:class:: template<class ordering = amgcl::reorder::cuthill_mckee<false>> \
               amgcl::adapter::reorder

   Reorders the matrix to reduce its bandwidth. Example:

   .. code-block:: cpp

      // Prepare the reordering:
      amgcl::adapter::reorder<> perm(A);

      // Create the solver using the reordered matrix:
      Solver solve(perm(A), prm);

      // Reorder the RHS and solve the system:
      solve(perm(rhs), x_ord);

      // Postprocess the solution vector to get the original ordering:
      perm.inverse(x_ord, x);

Eigen matrix
------------

Simply including ``<amgcl/adapter/eigen.hpp>`` allows to use Eigen sparse
matrices in AMGCL algorithm constructors. The Eigen matrix has to be stored
with the ``RowMajor`` ordering.

Epetra matrix
-------------

Including ``<amgcl/adapter/epetra.hpp>`` allows to use Trilinos Epetra
distributed sparse matrices in AMGCL MPI algorithm constructors.

uBlas matrix
------------

Including ``<amgcl/adapter/ublas.hpp>`` allows to use uBlas sparse
matrices in AMGCL algorithm constructors, and directly use uBlas vectors as the
RHS and solution arrays.

