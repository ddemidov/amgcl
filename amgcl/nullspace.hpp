#ifndef AMGCL_NULLSPACE_HPP
#define AMGCL_NULLSPACE_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
\file    amgcl/nullspace.hpp
\author  Denis Demidov <dennis.demidov@gmail.com>
\brief   Augments matrix with null-space vectors.
*/

#include <amgcl/backend/interface.hpp>
#include <boost/range.hpp>
#include <boost/type_traits.hpp>

namespace amgcl {
namespace nullspace {

template <class Matrix, class NullSpaceVectors>
struct matrix_with_ns {
    const Matrix           &A;
    const NullSpaceVectors &N;

    matrix_with_ns(const Matrix &A, const NullSpaceVectors &N)
        : A(A), N(N) {}
};

/// Returns matrix augmented with near null space vectors
/** The result may be used directly as a matrix by algorithms that do not
 * know/care about null space, or may be split into the matrix and the
 * null-space vectors otherwise
 */
template <class Matrix, class NullSpaceVectors>
matrix_with_ns<Matrix, NullSpaceVectors>
set_ns(const Matrix &A, const NullSpaceVectors &N) {
    return matrix_with_ns<Matrix, NullSpaceVectors>(A, N);
}

template <class T>
struct dummy { };

template <class Matrix>
struct vectors_impl {
    typedef typename math::rhs_of<
        typename backend::value_type<Matrix>::type
        >::type value_type;

    typedef dummy<value_type> type;
    static type get(const Matrix &A) { return dummy<value_type>(); }
};

template <class M, class NS>
struct vectors_impl< matrix_with_ns<M, NS> > {
    typedef const NS& type;
    static type get(const matrix_with_ns<M, NS> &A) { return A.N; }
};

/// Extract the nullspace vectors from matrix/nullspace vectors pair.
template <class Matrix>
typename vectors_impl<Matrix>::type vectors(const Matrix &A) {
    return vectors_impl<Matrix>::get(A);
}

template <class Matrix>
struct matrix_impl {
    typedef const Matrix& type;
    static type get(const Matrix &A) { return A; }
};

template <class M, class NS>
struct matrix_impl< matrix_with_ns<M, NS> > {
    typedef const M& type;
    static type get(const matrix_with_ns<M, NS> &A) { return A.A; }
};

/// Extract the matrix from matrix/nullspace vectors pair.
template <class Matrix>
typename matrix_impl<Matrix>::type matrix(const Matrix &A) {
    return matrix_impl<Matrix>::get(A);
}

template <class NS>
struct size_impl {
    static size_t get(const NS &V) {
        return boost::size(V);
    }
};

template <class value_type>
struct size_impl< dummy<value_type> > {
    static size_t get(const dummy<value_type> &V) { return 0; }
};

template <class Matrix>
size_t size(const Matrix &A) {
    return size_impl<Matrix>::get(A);
}

template <class NS>
struct value_impl {
    typedef
        typename boost::range_value<
            typename boost::range_value<
                typename boost::decay<NS>::type
            >::type
        >::type type;

    static type get(const NS &N, size_t vec, size_t i) {
        return N[vec][i];
    }
};

template <class V>
struct value_impl< dummy<V> > {
    typedef V type;

    static type get(const dummy<V> &NS, size_t vec, size_t i) {
        return math::zero<V>();
    }
};

template <class NS>
typename value_impl<NS>::type value(const NS &N, size_t vec, size_t i) {
    return value_impl<NS>::get(N, vec, i);
}

} // namespace nullspace

namespace backend {

template <class M, class NS>
struct value_type< nullspace::matrix_with_ns<M, NS> > : value_type<M> {};

template <class M, class NS>
struct rows_impl< nullspace::matrix_with_ns<M, NS> >
{
    static size_t get(const nullspace::matrix_with_ns<M, NS> &A) {
        return rows(A.A);
    }
};

template <class M, class NS>
struct cols_impl< nullspace::matrix_with_ns<M, NS> >
{
    static size_t get(const nullspace::matrix_with_ns<M, NS> &A) {
        return cols(A.A);
    }
};

template <class M, class NS>
struct nonzeros_impl< nullspace::matrix_with_ns<M, NS> >
{
    static size_t get(const nullspace::matrix_with_ns<M, NS> &A) {
        return nonzeros(A.A);
    }
};

template <class M, class NS>
struct row_iterator< nullspace::matrix_with_ns<M, NS> > : row_iterator<M> {};

template <class M, class NS>
struct row_begin_impl< nullspace::matrix_with_ns<M, NS> >
{
    static typename row_iterator<M>::type
    get(const nullspace::matrix_with_ns<M, NS> &A, size_t row) {
        return row_begin(A.A, row);
    }
};

namespace detail {

template <class M, class NS>
struct use_builtin_matrix_ops< nullspace::matrix_with_ns<M, NS> >
    : boost::true_type {};

} // namespace detail
} // namespace backend
} // namespace amgcl

#endif
