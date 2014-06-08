#ifndef AMGCL_BACKEND_INTERFACE_HPP
#define AMGCL_BACKEND_INTERFACE_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   amgcl/backend/interface.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Backend interface required for AMG.
 */

#include <cmath>

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <amgcl/util.hpp>

namespace amgcl {

/// Provided backends.
namespace backend {

/**
 * \defgroup backends Provided backends
 * \brief Backends implemented in AMGCL.
 *
 * A backend in AMGCL is a class that defines matrix and vector types together
 * with several operations on them, such as creation, matrix-vector products,
 * vector sums, inner products etc.  The AMG hierarchy is moved to the
 * specified backend upon construction. The solution phase then uses types and
 * operations defined in the backend. This enables transparent acceleration of
 * the solution phase with OpenMP, OpenCL, CUDA, or any other technologies.
 */

/**
 * \defgroup backend_interface Backend interface
 * \brief Backend interface specification.
 *
 * One has to specify these templates in order to define a new backend.
 */

/** \addtogroup backend_interface
 * @{
 */

/// Metafunction that returns value type of a matrix or a vector type.
template <class T, class Enable = void>
struct value_type {
    typedef typename T::value_type type;
};

/// Implementation for function returning the number of rows in a matrix.
/** \note Used in rows() */
template <class Matrix, class Enable = void>
struct rows_impl {
    typedef typename Matrix::ROWS_NOT_IMPLEMENTED type;
};

/// Implementation for function returning the number of columns in a matrix.
/** \note Used in cols() */
template <class Matrix, class Enable = void>
struct cols_impl {
    typedef typename Matrix::COLS_NOT_IMPLEMENTED type;
};

/// Implementation for function returning the number of nonzeros in a matrix.
/** \note Used in nonzeros() */
template <class Matrix, class Enable = void>
struct nonzeros_impl {
    typedef typename Matrix::NONZEROS_NOT_IMPLEMENTED type;
};

/// Metafunction returning the row iterator type for a matrix type.
/**
 * \note This only has to be implemented in the backend if support for serial
 * smoothers (Gauss-Seidel or ILU0) is required.
 */
template <class Matrix, class Enable = void>
struct row_iterator {
    typedef typename Matrix::ROW_ITERATOR_NOT_IMPLEMENTED type;
};

/// Implementation for function returning row iterator for a matrix.
/**
 * \note This only has to be implemented in the backend if support for serial
 * smoothers (Gauss-Seidel or ILU0) is required.
 * \note Used in row_begin()
 */
template <class Matrix, class Enable = void>
struct row_begin_impl {
    typedef typename Matrix::ROW_BEGIN_NOT_IMPLEMENTED type;
};

/// Implementation for matrix-vector product.
/** \note Used in spmv() */
template <class Matrix, class Vector, class Enable = void>
struct spmv_impl {
    typedef typename Matrix::SPMV_NOT_IMPLEMENTED type;
};

/// Implementation for residual error compuatation.
/** \note Used in residual() */
template <class Matrix, class Vector, class Enable = void>
struct residual_impl {
    typedef typename Matrix::RESIDUAL_NOT_IMPLEMENTED type;
};

/// Implementation for zeroing out a vector.
/** \note Used in clear() */
template <class Vector, class Enable = void>
struct clear_impl {
    typedef typename Vector::CLEAR_NOT_IMPLEMENTED type;
};

/// Implementation for vector copy.
/** \note Used in copy() */
template <class Vector, class Enable = void>
struct copy_impl {
    typedef typename Vector::COPY_NOT_IMPLEMENTED type;
};

/// Implementation for inner product.
/** \note Used in inner_product() */
template <class Vector, class Enable = void>
struct inner_product_impl {
    typedef typename Vector::INNER_PRODUCT_NOT_IMPLEMENTED type;
};

/// Implementation for vector norm.
/**
 * \note By default sqrt(inner_product(x, x)) is used as norm. One only has to
 *       specify this template if the default behavior has to be changed.
 * \note Used in norm()
 */
template <class Vector, class Enable = void>
struct norm_impl {
    static typename value_type<Vector>::type get(const Vector &x)
    {
        return sqrt( inner_product_impl<Vector>::get(x, x) );
    }
};

/// Implementation for linear combination of two vectors.
/** \note Used in axpby() */
template <class Vector, class Enable = void>
struct axpby_impl {
    typedef typename Vector::AXPBY_NOT_IMPLEMENTED type;
};

/// Implementation for linear combination of three vectors.
/** \note Used in axpbypcz() */
template <class Vector, class Enable = void>
struct axpbypcz_impl {
    typedef typename Vector::AXPBYPCZ_NOT_IMPLEMENTED type;
};

/// Implementation for element-wize vector product.
/** \note Used in vmul() */
template <class Vector, class Enable = void>
struct vmul_impl {
    typedef typename Vector::VMUL_NOT_IMPLEMENTED type;
};

/** @} */

/// Returns the number of rows in a matrix.
template <class Matrix>
size_t rows(const Matrix &matrix) {
    return rows_impl<Matrix>::get(matrix);
}

/// Returns the number of columns in a matrix.
template <class Matrix>
size_t cols(const Matrix &matrix) {
    return cols_impl<Matrix>::get(matrix);
}

/// Returns the number of nonzeros in a matrix.
template <class Matrix>
size_t nonzeros(const Matrix &matrix) {
    return nonzeros_impl<Matrix>::get(matrix);
}

/// Returns row iterator for a matrix.
template <class Matrix>
typename row_iterator<Matrix>::type
row_begin(const Matrix &matrix, size_t row) {
    return row_begin_impl<Matrix>::get(matrix, row);
}

/// Performs matrix-vector product.
/**
 * \f[y = \alpha A x + \beta y.\f]
 */
template <class Matrix, class Vector>
void spmv(typename value_type<Matrix>::type alpha, const Matrix &A,
        const Vector &x, typename value_type<Matrix>::type beta, Vector &y)
{
    spmv_impl<Matrix, Vector>::apply(alpha, A, x, beta, y);
}

/// Computes residual error.
/**
 * \f[r = rhs - Ax.\f]
 */
template <class Matrix, class Vector>
void residual(const Vector &rhs, const Matrix &A, const Vector &x, Vector &r)
{
    residual_impl<Matrix, Vector>::apply(rhs, A, x, r);
}

/// Zeros out a vector.
template <class Vector>
void clear(Vector &x)
{
    clear_impl<Vector>::apply(x);
}

/// Vector copy.
template <class Vector>
void copy(const Vector &x, Vector &y)
{
    copy_impl<Vector>::apply(x, y);
}

/// Computes inner product of two vectors.
template <class Vector>
typename value_type<Vector>::type
inner_product(const Vector &x, const Vector &y)
{
    return inner_product_impl<Vector>::get(x, y);
}

/// Returns norm of a vector.
template <class Vector>
typename value_type<Vector>::type norm(const Vector &x)
{
    return norm_impl<Vector>::get(x);
}

/// Computes linear combination of two vectors.
/**
 * \f[y = ax + by.\f]
 */
template <class Vector>
void axpby(typename value_type<Vector>::type a, Vector const &x,
           typename value_type<Vector>::type b, Vector       &y
           )
{
    axpby_impl<Vector>::apply(a, x, b, y);
}

/// Computes linear combination of three vectors.
/**
 * \f[z = ax + by + cz.\f]
 */
template <class Vector>
void axpbypcz(
        typename value_type<Vector>::type a, Vector const &x,
        typename value_type<Vector>::type b, Vector const &y,
        typename value_type<Vector>::type c, Vector       &z
        )
{
    axpbypcz_impl<Vector>::apply(a, x, b, y, c, z);
}

/// Computes element-wize vector product.
/**
 * \f[z = \alpha xy + \beta z.\f]
 */
template <class Vector>
void vmul(
        typename value_type<Vector>::type alpha,
        const Vector &x, const Vector &y,
        typename value_type<Vector>::type beta,
        Vector &z
        )
{
    vmul_impl<Vector>::apply(alpha, x, y, beta, z);
}

} // namespace backend
} // namespace amgcl


#endif
