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
namespace backend {

template <class T, class Enable = void>
struct value_type {
    typedef typename T::value_type type;
};

/// Number of rows in a matrix (default implementation).
/** \note Does nothing, should be specialized for the concrete type.
 */
template <class Matrix, class Enable = void>
struct rows_impl {
    static size_t get(const Matrix&) {
        precondition(false, "nrows() not implemented");
        return 0;
    }
};

/// Returns number of rows in a matrix.
template <class Matrix>
size_t rows(const Matrix &matrix) {
    return rows_impl<Matrix>::get(matrix);
}

/// Number of columns in a matrix (default implementation).
/** \note Does nothing, should be specialized for the concrete type.
 */
template <class Matrix, class Enable = void>
struct cols_impl {
    static size_t get(const Matrix&) {
        precondition(false, "ncols() not implemented");
        return 0;
    }
};

/// Returns number of columns in a matrix.
template <class Matrix>
size_t cols(const Matrix &matrix) {
    return cols_impl<Matrix>::get(matrix);
}

/// Number of nonzero values in a matrix (default implementation).
/** \note Does nothing, should be specialized for the concrete type.
 */
template <class Matrix, class Enable = void>
struct nonzeros_impl {
    static size_t get(const Matrix&) {
        precondition(false, "nonzeros() not implemented");
        return 0;
    }
};

/// Returns number of nonzero values in a matrix.
template <class Matrix>
size_t nonzeros(const Matrix &matrix) {
    return nonzeros_impl<Matrix>::get(matrix);
}

/// Row iterator type (default implementation).
/** \note Does nothing, should be specialized for the concrete type.
 */
template <class Matrix, class Enable = void>
struct row_iterator {
    typedef typename Matrix::ROW_ITERATOR_IS_NOT_IMPLEMENTED type;
};

template <class Matrix, class Enable = void>
struct row_begin_impl {
    static typename row_iterator<Matrix>::type
    get(const Matrix&, size_t) {
        precondition(false, "row_begin() not implemented");
    }
};

template <class Matrix>
typename row_iterator<Matrix>::type
row_begin(const Matrix &matrix, size_t row) {
    return row_begin_impl<Matrix>::get(matrix, row);
}

template <class Matrix, class Vector, class Enable = void>
struct spmv_impl {
    typedef typename value_type<Matrix>::type real;

    static void apply(real, const Matrix&, const Vector&, real, Vector&)
    {
        precondition(false, "spmv() not implemented");
    }
};

template <class Matrix, class Vector>
void spmv(typename value_type<Matrix>::type alpha, const Matrix &A,
        const Vector &x, typename value_type<Matrix>::type beta, Vector &y)
{
    spmv_impl<Matrix, Vector>::apply(alpha, A, x, beta, y);
}

template <class Matrix, class Vector, class Enable = void>
struct residual_impl {
    static void apply(const Vector&, const Matrix&, const Vector&, Vector&)
    {
        precondition(false, "residual() not implemented");
    }
};

template <class Matrix, class Vector>
void residual(const Vector &rhs, const Matrix &A, const Vector &x, Vector &r)
{
    residual_impl<Matrix, Vector>::apply(rhs, A, x, r);
}

template <class Vector, class Enable = void>
struct clear_impl {
    static void apply(Vector&)
    {
        precondition(false, "clear() not implemented");
    }
};

template <class Vector>
void clear(Vector &x)
{
    clear_impl<Vector>::apply(x);
}

template <class Vector, class Enable = void>
struct inner_product_impl {
    static typename value_type<Vector>::type
    get(const Vector&, const Vector&)
    {
        precondition(false, "inner_product() not implemented");
        return typename value_type<Vector>::type();
    }
};

template <class Vector>
typename value_type<Vector>::type
inner_product(const Vector &x, const Vector &y)
{
    return inner_product_impl<Vector>::get(x, y);
}

template <class Vector, class Enable = void>
struct norm_impl {
    static typename value_type<Vector>::type get(const Vector &x)
    {
        return sqrt( inner_product(x, x) );
    }
};

template <class Vector>
typename value_type<Vector>::type norm(const Vector &x)
{
    return norm_impl<Vector>::get(x);
}

template <class Vector, class Enable = void>
struct axpby_impl {
    typedef typename value_type<Vector>::type val_type;

    static void apply(val_type, const Vector&, val_type, Vector&)
    {
        precondition(false, "axpby() not implemented");
    }
};

template <class Vector>
void axpby(typename value_type<Vector>::type a, Vector const &x,
           typename value_type<Vector>::type b, Vector       &y
           )
{
    axpby_impl<Vector>::apply(a, x, b, y);
}

template <class Vector, class Enable = void>
struct vmul_impl {
    typedef typename value_type<Vector>::type val_type;

    static void apply(val_type, const Vector&, const Vector&, val_type, Vector&)
    {
        precondition(false, "axpby() not implemented");
    }
};

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

template <class Vector, class Enable = void>
struct copy_impl {
    static void apply(const Vector&, Vector&)
    {
        precondition(false, "copy() not implemented");
    }
};

template <class Vector>
void copy(const Vector &x, Vector &y)
{
    copy_impl<Vector>::apply(x, y);
}

} // namespace backend
} // namespace amgcl


#endif
