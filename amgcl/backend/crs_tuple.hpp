#ifndef AMGCL_BACKEND_CRS_TUPLE_HPP
#define AMGCL_BACKEND_CRS_TUPLE_HPP

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
 * \file   amgcl/backend/crs_tuple.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Specify sparse matrix as a tuple of CRS arrays.
 *
 * Example:
 * \code
 * // Adapt STL containers:
 * std::vector<int>    ptr;
 * std::vector<int>    col;
 * std::vector<double> val;
 *
 * AMG amg( boost::tie(n, n, val, col, ptr) );
 *
 * // Adapt raw arrays:
 * int    *ptr;
 * int    *col;
 * double *val;
 *
 * AMG amg(boost::make_tuple(n, n,
 *                           boost::make_iterator_range(val, val + nnz),
 *                           boost::make_iterator_range(col, col + nnz),
 *                           boost::make_iterator_range(ptr, ptr + n + 1)
 *                           ) );
 * \endcode
 */

#include <vector>
#include <numeric>

#include <boost/tuple/tuple.hpp>
#include <boost/range.hpp>
#include <boost/type_traits.hpp>

#include <amgcl/util.hpp>
#include <amgcl/backend/interface.hpp>

namespace amgcl {
namespace backend {

//---------------------------------------------------------------------------
// Specialization of matrix interface
//---------------------------------------------------------------------------
template < typename NR, typename NC, typename VRng, typename CRng, typename PRng >
struct value_type< boost::tuple<NR, NC, VRng, CRng, PRng> >
{
    typedef
        typename boost::range_value<
                    typename boost::decay<VRng>::type
                 >::type
        type;
};

template < typename NR, typename NC, typename VRng, typename CRng, typename PRng >
struct rows_impl< boost::tuple<NR, NC, VRng, CRng, PRng> >
{
    static size_t get(
            const boost::tuple<NR, NC, VRng, CRng, PRng> &A
            )
    {
        return boost::get<0>(A);
    }
};

template < typename NR, typename NC, typename VRng, typename CRng, typename PRng >
struct cols_impl< boost::tuple<NR, NC, VRng, CRng, PRng> >
{
    static size_t get(
            const boost::tuple<NR, NC, VRng, CRng, PRng> &A
            )
    {
        return boost::get<1>(A);
    }
};

template < typename NR, typename NC, typename VRng, typename CRng, typename PRng >
struct nonzeros_impl< boost::tuple<NR, NC, VRng, CRng, PRng> >
{
    static size_t get(
            const boost::tuple<NR, NC, VRng, CRng, PRng> &A
            )
    {
        return *( boost::begin(boost::get<4>(A)) + boost::get<0>(A) );
    }
};

template < typename NR, typename NC, typename VRng, typename CRng, typename PRng >
struct row_iterator< boost::tuple<NR, NC, VRng, CRng, PRng> >
{
    class type {
        public:
            typedef
                typename boost::range_value<
                            typename boost::decay<CRng>::type
                         >::type
                col_type;
            typedef
                typename boost::range_value<
                            typename boost::decay<VRng>::type
                         >::type
                val_type;

            type(const boost::tuple<NR, NC, VRng, CRng, PRng> &A,
                 size_t row)
            {
                typedef
                    typename boost::range_value<
                                typename boost::decay<PRng>::type
                             >::type
                    ptr_type;

                ptr_type row_begin = *(boost::begin( boost::get<4>(A) ) + row);
                ptr_type row_end   = *(boost::begin( boost::get<4>(A) ) + row + 1);

                m_col = boost::begin( boost::get<3>(A) ) + row_begin;
                m_end = boost::begin( boost::get<3>(A) ) + row_end;
                m_val = boost::begin( boost::get<2>(A) ) + row_begin;
            }

            operator bool() const {
                return m_col != m_end;
            }

            type& operator++() {
                ++m_col;
                ++m_val;
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
                            typename boost::decay<VRng>::type
                         >::type
                val_iterator;

            typedef
                typename boost::range_const_iterator<
                            typename boost::decay<CRng>::type
                         >::type
                col_iterator;

            col_iterator m_col;
            col_iterator m_end;
            val_iterator m_val;
    };
};

template < typename NR, typename NC, typename VRng, typename CRng, typename PRng >
struct row_begin_impl< boost::tuple<NR, NC, VRng, CRng, PRng> >
{
    typedef boost::tuple<NR, NC, VRng, CRng, PRng> Matrix;
    static typename row_iterator<Matrix>::type
    get(const Matrix &matrix, size_t row) {
        return typename row_iterator<Matrix>::type(matrix, row);
    }
};

} // namespace backend
} // namespace amgcl

#endif
