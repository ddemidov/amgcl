Matrix adapters
---------------

Matrix adapters in AMGCL allow to construct a solver from some common matrix
formats. Internally, the CRS_ format is used, but it is easy to adapt any
matrix format that allows row-wise access to the nonzero matrix values. An
example of creating an adapter is provided in :doc:`custom_adapter`.

Boost tuple adapter
########################

``#include`` `\<amgcl/adapter/crs_tuple.hpp>`_

The Boost tuple adapter allows to use a ``boost::tuple`` of a matrix size and
its three CRS_ format components (row pointer array, column indices array, and
values array) as input matrix to AMGCL solvers. The arrays are allowed to be in
any format recognized by the Boost.Range_ library as a random access range.
Common examples are STL vectors and Boost `iterator ranges`_.

Example:

.. code-block:: cpp

    // boost::tie creates a tuple of references, which avoids copying.
    Solver solve( boost::tie(n, ptr, col, val) );

    // A (cheap) copy is required when iterator ranges are created on the fly:
    Solver solve( boost::make_tuple(
        n,
        boost::make_iterator_range(ptr.data(), ptr.data() + ptr.size()),
        boost::make_iterator_range(col.data(), col.data() + col.size()),
        boost::make_iterator_range(val.data(), val.data() + val.size())
        ) );

Boost.uBLAS adapter
###################

``#include`` `\<amgcl/adapter/ublas.hpp>`_

The Boost.uBLAS_ adapter allows to use uBLAS sparse matrices as input to AMGCL
solvers. It also allows to use uBLAS dense vectors with
:cpp:class:`amgcl::backend::builtin`.

Example:

.. code-block:: cpp

    namespace ublas = boost::numeric::ublas;

    ublas::compressed_matrix<double> A;
    ...
    Solver solve(A);

    ublas::vector<double> rhs, x;
    ...
    solve(rhs, x);

Zero copy adapter
#################

``#include`` `\<amgcl/adapter/zero_copy.hpp>`_

In general, AMGCL copies the adapted input matrix into its internal structures,
so that the matrix may be safely destroyed or reused as soon as the solver
setup is complete. However, the memory overhead of the copying may be too
large, especially for large problems that eat up almost all of available RAM.
The zero copy adapter allows to use raw pointers to CRS arrays as input matrix
for MAGCL solvers. The data from the arrays is never copied during setup, and
the user has to make sure the arrays stay alive long enough. However, unless
the backend used is :cpp:class:`amgcl::backend::builtin`, the input matrix will
be copied into the backend structures when the setup is finished. This would
still allow to save some memory in case of GPGPU backends.

The one requirement is that the integer types stored in row pointers and column
indices arrays have to be binary compatible with ``ptrdiff_t``, and the value
type has to be the value type of the backend.

Example:

.. code-block:: cpp

    Solver solve( amgcl::adapter::zero_copy(n, &ptr[0], &col[0], &val[0]) );

.. _CRS: http://netlib.org/linalg/html_templates/node91.html

.. _Boost.Range: http://www.boost.org/doc/libs/release/libs/range/
.. _iterator ranges: http://www.boost.org/doc/libs/release/libs/range/doc/html/range/reference/utilities/iterator_range.html
.. _Boost.uBLAS: http://www.boost.org/doc/libs/release/libs/numeric/ublas/

.. _\<amgcl/adapter/crs_tuple.hpp>: https://github.com/ddemidov/amgcl/blob/master/amgcl/adapter/crs_tuple.hpp
.. _\<amgcl/adapter/ublas.hpp>: https://github.com/ddemidov/amgcl/blob/master/amgcl/adapter/ublas.hpp
.. _\<amgcl/adapter/zero_copy.hpp>: https://github.com/ddemidov/amgcl/blob/master/amgcl/adapter/zero_copy.hpp
