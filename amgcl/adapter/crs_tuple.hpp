#ifndef AMGCL_ADAPTER_CRS_TUPLE_HPP
#define AMGCL_ADAPTER_CRS_TUPLE_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
\file    amgcl/adapter/crs_tuple.hpp
\author  Denis Demidov <dennis.demidov@gmail.com>
\brief   Specify sparse matrix as a tuple of CRS arrays.
\ingroup adapters

Example:
\code
// Adapt STL containers:
std::vector<int>    ptr;
std::vector<int>    col;
std::vector<double> val;

AMG amg( std::tie(n, ptr, col, val) );

// Adapt raw arrays:
int    *ptr;
int    *col;
double *val;

AMG amg(std::make_tuple(n,
                          boost::make_iterator_range(ptr, ptr + n + 1),
                          boost::make_iterator_range(col, col + ptr[n]),
                          boost::make_iterator_range(val, val + ptr[n])
                          ) );
\endcode
*/

/**
 * \defgroup adapters Matrix adapters
 * \brief Adapters for variuos sparse matrix formats.
 */

#include <vector>
#include <numeric>

#include <tuple>
#include <boost/range/const_iterator.hpp>
#include <boost/range/value_type.hpp>
#include <boost/range/size.hpp>
#include <type_traits>

#include <amgcl/util.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/backend/detail/matrix_ops.hpp>

namespace amgcl {
namespace backend {

//---------------------------------------------------------------------------
// Specialization of matrix interface
//---------------------------------------------------------------------------
template < typename N, typename PRng, typename CRng, typename VRng >
struct value_type< std::tuple<N, PRng, CRng, VRng> >
{
    typedef
        typename boost::range_value<
                    typename std::decay<VRng>::type
                 >::type
        type;
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct rows_impl< std::tuple<N, PRng, CRng, VRng> >
{
    static size_t get(
            const std::tuple<N, PRng, CRng, VRng> &A
            )
    {
        return std::get<0>(A);
    }
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct cols_impl< std::tuple<N, PRng, CRng, VRng> >
{
    static size_t get(
            const std::tuple<N, PRng, CRng, VRng> &A
            )
    {
        return std::get<0>(A);
    }
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct nonzeros_impl< std::tuple<N, PRng, CRng, VRng> >
{
    static size_t get(
            const std::tuple<N, PRng, CRng, VRng> &A
            )
    {
        return std::get<1>(A)[std::get<0>(A)];
    }
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct row_iterator< std::tuple<N, PRng, CRng, VRng> >
{
    class type {
        public:
            typedef
                typename boost::range_value<
                            typename std::decay<CRng>::type
                         >::type
                col_type;
            typedef
                typename boost::range_value<
                            typename std::decay<VRng>::type
                         >::type
                val_type;

            type(const std::tuple<N, PRng, CRng, VRng> &A, size_t row)
                : m_col(boost::begin(std::get<2>(A)))
                , m_end(boost::begin(std::get<2>(A)))
                , m_val(boost::begin(std::get<3>(A)))
            {
                typedef
                    typename boost::range_value<
                                typename std::decay<PRng>::type
                             >::type
                    ptr_type;

                ptr_type row_begin = std::get<1>(A)[row];
                ptr_type row_end   = std::get<1>(A)[row + 1];

                m_col += row_begin;
                m_end += row_end;
                m_val += row_begin;
            }

            operator bool() const {
                return m_col != m_end;
            }

            type& operator++() {
                ++m_col;
                ++m_val;
                return *this;
            }

            col_type col() const {
                return *m_col;
            }

            val_type value() const {
                return *m_val;
            }

        private:
            typedef
                typename boost::range_const_iterator<
                            typename std::decay<VRng>::type
                         >::type
                val_iterator;

            typedef
                typename boost::range_const_iterator<
                            typename std::decay<CRng>::type
                         >::type
                col_iterator;

            col_iterator m_col;
            col_iterator m_end;
            val_iterator m_val;
    };
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct row_begin_impl< std::tuple<N, PRng, CRng, VRng> >
{
    typedef std::tuple<N, PRng, CRng, VRng> Matrix;
    static typename row_iterator<Matrix>::type
    get(const Matrix &matrix, size_t row) {
        return typename row_iterator<Matrix>::type(matrix, row);
    }
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct row_nonzeros_impl< std::tuple<N, PRng, CRng, VRng> > {
    typedef std::tuple<N, PRng, CRng, VRng> Matrix;

    static size_t get(const Matrix &A, size_t row) {
        return std::get<1>(A)[row + 1] - std::get<1>(A)[row];
    }
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct ptr_data_impl< std::tuple<N, PRng, CRng, VRng> > {
    typedef std::tuple<N, PRng, CRng, VRng> Matrix;
    typedef
        typename boost::range_value<
                    typename std::decay<PRng>::type
                 >::type
        ptr_type;
    typedef const ptr_type* type;
    static type get(const Matrix &A) {
        return &std::get<1>(A)[0];
    }
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct col_data_impl< std::tuple<N, PRng, CRng, VRng> > {
    typedef std::tuple<N, PRng, CRng, VRng> Matrix;
    typedef
        typename boost::range_value<
                    typename std::decay<CRng>::type
                 >::type
        col_type;
    typedef const col_type* type;
    static type get(const Matrix &A) {
        return &std::get<2>(A)[0];
    }
};

template < typename N, typename PRng, typename CRng, typename VRng >
struct val_data_impl< std::tuple<N, PRng, CRng, VRng> > {
    typedef std::tuple<N, PRng, CRng, VRng> Matrix;
    typedef
        typename boost::range_value<
                    typename std::decay<VRng>::type
                 >::type
        val_type;
    typedef const val_type* type;
    static type get(const Matrix &A) {
        return &std::get<3>(A)[0];
    }
};

namespace detail {

template < typename N, typename PRng, typename CRng, typename VRng >
struct use_builtin_matrix_ops< std::tuple<N, PRng, CRng, VRng> >
    : std::true_type
{};

} // namespace detail

} // namespace backend
} // namespace amgcl

#endif
