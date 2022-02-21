Relaxation
==========

A relaxation method or a smoother is used on each level of the AMG hierarchy
during solution phase.

Damped Jacobi
-------------

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::damped_jacobi

   .. rubric:: Include ``<amgcl/relaxation/damped_jacobi.hpp>``

   The damped Jacobi relaxation

   .. cpp:class:: params

      Damped Jacobi relaxation parameters

      .. cpp:type:: typename Backend::value_type value_type

         The value type of the system matrix

      .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

         The scalar type corresponding to the value type. For example, when the
         value type is ``std::complex<double>``, then the scalar type is
         ``double``.

      .. cpp:member:: scalar_type damping = 0.72

         The damping factor


Gauss-Seidel
------------

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::gauss_seidel

   .. rubric:: Include ``<amgcl/relaxation/gauss_seidel.hpp>``

   The Gauss-Seidel relaxation. The relaxation is only available for the
   backends where the matrix supports row-wise iteration over its non-zero
   values.

   .. cpp:class:: params

      Gauss-Seidel relaxation parameters

      .. cpp:member:: bool serial = false

         Use the serial version of the algorithm

Chebyshev
---------

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::chebyshev

   .. rubric:: Include ``<amgcl/relaxation/chebyshev.hpp>``

   Chebyshev iteration is an iterative method for the solution of a system of
   linear equations, and unlike Jacobi, it is not a stationary method. However,
   it does not require inner products like many other nonstationary methods
   (most Krylov methods). These inner products can be a performance bottleneck
   on certain distributed memory architectures. Furthermore, Chebyshev
   iteration is, like Jacobi, easier to parallelize than for instance
   Gauss–Seidel smoothers. The Chebyshev iteration requires some information
   about the spectrum of the matrix. For symmetric matrices, it needs an upper
   bound for the largest eigenvalue and a lower bound for the smallest
   eigenvalue [GhKK12]_.

   .. cpp:class:: params

      Chebyshev relaxation parameters

      .. cpp:member:: unsigned degree = 5
         
         The degree of Chebyshev polynomial

      .. cpp:member:: float higher = 1.0

         The highest eigen value safety upscaling.
         use boosting factor for a more conservative upper bound estimate [ABHT03]_.

      .. cpp:member:: float lower = 1.0/30

         Lowest-to-highest eigen value ratio.

      .. cpp:member:: int power_iters = 0

         The number of power iterations to apply for the spectral radius
         estimation. When 0, use Gershgorin disk theorem to estimate
         the spectral radius.

      .. cpp:member:: bool scale = false

         Scale the system matrix


Incomplete LU relaxation
------------------------

The incomplete LU factorization process computes a sparse lower triangular
matrix :math:`L` and a sparse upper triangular matrix :math:`U` so that the
residual matrix :math:`R = LU - A` satisfies certain constraints, such as
having zero entries in some locations. The relaxations in this section use
various approaches to computation of the triangular factors :math:`L` and
:math:`U`, but share the triangular system solution implementation required
in order to apply the relaxation. The parameters for the triangular solution
algorithm are defined as follows:

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::detail::ilu_solve

   For the builtin OpenMP backend the incomplete triangular
   factors are solved using the OpenMP-parallel level scheduling
   approach. For the GPGPU backends, the triangular factors are solved
   approximately, using multiple damped Jacobi iterations [ChPa15]_.

   .. cpp:class:: params

      .. cpp:member:: bool serial = false

         Use the serial version of the algorithm. This parameter is only
         used with the :cpp:class:`amgcl::backend::builtin` backend.

      .. cpp:member:: unsigned iters = 2

         The number of Jacobi iterations to approximate the triangular
         system solution. This parameter is only used with GPGPU backends.
      
      .. cpp:member:: scalar_type damping = 1.0

         The damping factor for the triangular solve approximation. This
         parameter is only used with GPGPU backends.

ILU0
^^^^
.. cpp:class:: template <class Backend> \
               amgcl::relaxation::ilu0

   .. rubric:: Include ``<amgcl/relaxation/ilu0.hpp>``

   The incomplete LU factorization with zero fill-in [Saad03]_. The zero
   pattern for the triangular factors :math:`L` and :math:`U` is taken to be
   exactly the zero pattern of the system matrix :math:`A`.

   .. cpp:class:: params

      ILU0 relaxation parameters

      .. cpp:type:: typename Backend::value_type value_type

         The value type of the system matrix

      .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

         The scalar type corresponding to the value type. For example, when the
         value type is ``std::complex<double>``, then the scalar type is
         ``double``.

      .. cpp:member:: scalar_type damping = 1.0

         The damping factor

      .. cpp:member:: typename amgcl::relaxation::detail::ilu_solve<Backend>::params solve

         The parameters for the triangular factor solver


ILUK
^^^^

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::iluk

   .. rubric:: Include ``<amgcl/relaxation/iluk.hpp>``

   The ILU(k) relaxation.

   The incomplete LU factorization with the level of fill-in [Saad03]_. The
   accuracy of the ILU0 incomplete factorization may be insufficient to yield
   an adequate rate of convergence. More accurate incomplete LU factorizations
   are often more efficient as well as more reliable. These more accurate
   factorizations will differ from ILU(0) by allowing some fill-in. Thus,
   ILUK(k) keeps the 'k-th order fill-ins' [Saad03]_.

   The ILU(1) factorization results from taking the zero pattern for triangular
   factors to be the zero pattern of the product :math:`L_0 U_0` of the factors
   :math:`L_0`, :math:`U_0` obtained from ILU(0). This process is repeated to
   obtain the higher level of fill-in factorizations.

   .. cpp:class:: params

      ILUK relaxation parameters

      .. cpp:type:: typename Backend::value_type value_type

         The value type of the system matrix

      .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

         The scalar type corresponding to the value type. For example, when the
         value type is ``std::complex<double>``, then the scalar type is
         ``double``.

      .. cpp:member:: int k = 1

         The level of fill-in

      .. cpp:member:: scalar_type damping = 1.0

         The damping factor

      .. cpp:member:: typename amgcl::relaxation::detail::ilu_solve<Backend>::params solve

         The parameters for the triangular factor solver

ILUP
^^^^

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::ilup

   .. rubric:: Include ``<amgcl/relaxation/ilup.hpp>``

   The ILUP(k) relaxation.

   This variant of the ILU relaxation is similar to ILUK, but differs in the
   way the zero pattern for the triangular factors is determined. Instead of
   the recursive definition using the product :math:`LU` of the factors from
   the previous level of fill-in, ILUP uses the powers of the boolean matrix
   :math:`S` sharing the zero pattern with the system matrix :math:`A`
   [MiKu03]_. ILUP(0) coinsides with ILU0, ILUP(1) has the same zero pattern as
   :math:`S^2`, etc.

   .. cpp:class:: params

      ILUP relaxation parameters

      .. cpp:type:: typename Backend::value_type value_type

         The value type of the system matrix

      .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

         The scalar type corresponding to the value type. For example, when the
         value type is ``std::complex<double>``, then the scalar type is
         ``double``.

      .. cpp:member:: int k = 1

         The level of fill-in

      .. cpp:member:: scalar_type damping = 1.0

         The damping factor

      .. cpp:member:: typename amgcl::relaxation::detail::ilu_solve<Backend>::params solve

         The parameters for the triangular factor solver

ILUT
^^^^

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::ilut

   .. rubric:: Include ``<amgcl/relaxation/ilut.hpp>``

   The :math:`\mathrm{ILUT}(p,\tau)` relaxation.

   Incomplete factorizations which rely on the levels of fill are blind to
   numerical values because elements that are dropped depend only on the
   structure of A. This can cause some difficulties for realistic problems that
   arise in many applications. A few alternative methods are available which
   are based on dropping elements in the Gaussian elimination process according
   to their magnitude rather than their locations. With these techniques, the
   zero pattern P is determined dynamically.

   A generic ILU algorithm with threshold can be derived from the IKJ version
   of Gaussian elimination by including a set of rules for dropping small
   elements.  In the factorization :math:`\mathrm{ILUT}(p,\tau)`, the following rule is
   used:

   1. an element is dropped (i.e., replaced by zero) if it is less than the
      relative tolerance :math:`\tau_i` obtained by multiplying :math:`\tau` by
      the original 2-norm of the i-th row.
   2. Only the :math:`p l_i` largest elements are kept in the :math:`L` part of the
      row and the :math:`p u_i` largest elements in the :math:`U` part of the row
      in addition to the diagonal element, which is always kept. :math:`l_i`
      and :math:`u_i` are the number of nonzero elements in the i-th row of the
      system matrix :math:`A` below and above the diagonal.

   .. cpp:class:: params

      ILUT relaxation parameters

      .. cpp:type:: typename Backend::value_type value_type

         The value type of the system matrix

      .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

         The scalar type corresponding to the value type. For example, when the
         value type is ``std::complex<double>``, then the scalar type is
         ``double``.

      .. cpp:member:: scalar_type p = 2

         The fill factor

      .. cpp:member:: scalar_type tau = 1e-2

         The minimum magnitude of non-zero elements relative to the current row norm.

      .. cpp:member:: scalar_type damping = 1.0

         The damping factor

      .. cpp:member:: typename amgcl::relaxation::detail::ilu_solve<Backend>::params solve

         The parameters for the triangular factor solver

Sparse Approximate Inverse relaxation
-------------------------------------

Sparse approximate inverse (SPAI) smoothers based on the SPAI algorithm by
Grote and Huckle [GrHu97]_. The SPAI algorithm computes an approximate inverse
:math:`M` explicitly by minimizing :math:`I - MA` in the Frobenius norm. Both
the computation of :math:`M` and its application as a smoother are inherently
parallel. Since an effective sparsity pattern of :math:`M` is in general
unknown a priori, the computation cost can be greately reduced by choosing an a
priori sparsity pattern for :math:`M`. For SPAI-0 and SPAI-1 the sparsity
pattern of :math:`M` is fixed: :math:`M` is diagonal for SPAI-0, whereas for
SPAI-1 the sparsity pattern of :math:`M` is that of :math:`A` [BrGr02]_.

SPAI0
^^^^^

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::spai0

   .. rubric:: Include ``<amgcl/relaxation/spai0.hpp>``

   The SPAI-0 variant of the sparse approximate inverse smother [BrGr02]_.

   .. cpp:class:: params

      The SPAI-0 has no parameters

SPAI1
^^^^^

.. cpp:class:: template <class Backend> \
               amgcl::relaxation::spai1

   .. rubric:: Include ``<amgcl/relaxation/spai1.hpp>``

   The SPAI-1 variant of the sparse approximate inverse smother [BrGr02]_.

   .. cpp:class:: params

      The SPAI-1 has no parameters

Scalar to Block convertor
-------------------------

.. cpp:class:: template <class BlockBackend, template <class> class Relax> \
               amgcl::relaxation::as_block
    
   .. rubric:: Include ``<amgcl/relaxation/as_block.hpp>``

   Wrapper for the specified relaxation. Converts the input matrix from scalar
   to block format before constructing an amgcl smoother. See the
   :doc:`/tutorial/Nullspace` tutorial.

   .. cpp:class:: template <class Backend>
                  type

      The resulting relaxation class.
