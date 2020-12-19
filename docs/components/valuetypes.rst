Value Types
===========

The value type concept allows to generalize AMGCL algorithms for the systems
with complex or non-scalar coeffiecients. A value type defines a number of
overloads for common math operations, and is used as a template parameter for a
backend. Most often, a value type is simply a builtin ``double`` or ``float``
atomic value, but it is also possible to use ``std::complex<T>``, or small
statically sized matrices when the system matrix has a block structure. The
latter may decrease the setup time and the overall memory footprint, increase
cache locality, or improve convergence ratio.

Value types are used during both the setup and the solution phases. Common
value type operations are defined in ``amgcl::math`` namespace, similar to how
backend operations are defined in ``amgcl::backend``. Examples of such
operations are ``amgcl::math::norm()`` or ``amgcl::math::adjoint()``.
Arithmetic operations like multiplication or addition are defined as operator
overloads.  AMGCL algorithms at the lowest level are expressed in terms of the
value type interface, which makes it possible to switch precision of the
algorithms, or move to complex values simply by adjusting the template parameter
of the selected backend.

The generic implementation of the value type operations also makes it possible
to use efficient third party implementations of the block value arithmetics.
For example, using statically sized Eigen_ matrices instead of the builtin
``amgcl::static_matrix`` as block value type may improve performance in case of
relatively large blocks, since the Eigen_ library supports SIMD vectorization.

Scalar values
-------------

All backends support ``float`` and ``double`` as value type. CPU-based backends
(e.g. :cpp:class:`amgcl::backend::builtin<VT>`) may also use ``long double``.
The use of non-trivial value types depends on whether the value type is
supported by the selected backend.

.. _Eigen: http://eigen.tuxfamily.org

Complex values
--------------

.. rubric:: Data type: ``std::complex<T>``

.. rubric:: Include:

- ``<amgcl/value_type/complex.hpp>``

.. rubric:: Supported by backends:

- :cpp:class:`amgcl::backend::builtin`
- :cpp:class:`amgcl::backend::vexcl`
- :cpp:class:`amgcl::backend::eigen`
- :cpp:class:`amgcl::backend::blaze`

Statically sized matrices
-------------------------

.. rubric:: Data type: ``amgcl::static_matrix<T,N,N>``

.. rubric:: Include:

- ``<amgcl/value_type/static_matrix.hpp>``
- ``<amgcl/backend/vexcl_static_matrix.hpp>`` (in case VexCL is used as the
  backend)

.. rubric:: Supported by backends:

- :cpp:class:`amgcl::backend::builtin`
- :cpp:class:`amgcl::backend::vexcl`

Eigen static matrices
---------------------

.. rubric:: Data type: ``Eigen::Matrix<T,N,N>``

.. rubric:: Include:

- ``<amgcl/value_type/eigen.hpp>``

.. rubric:: Supported by backends:

- :cpp:class:`amgcl::backend::builtin`
- :cpp:class:`amgcl::backend::eigen`
