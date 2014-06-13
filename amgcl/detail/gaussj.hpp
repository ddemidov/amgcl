#ifndef AMGCL_DETAIL_GAUSSJ_HPP
#define AMGCL_DETAIL_GAUSSJ_HPP

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
 * \file   amgcl/detail/gaussj.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Inverse of a dense matrix with Gauss-Jordan algorithm.
 */

#include <vector>
#include <boost/multi_array.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace detail {

    template <typename value_type>
    static void gaussj(size_t n, value_type *data) {
        boost::multi_array_ref<value_type, 2> a(data, boost::extents[n][n]);

        std::vector<size_t>  idxc(n);
        std::vector<size_t>  idxr(n);
        std::vector<char>    ipiv(n, false);

        for(size_t i = 0; i < n; ++i) {
            size_t irow = 0, icol = 0;

            value_type big = 0;
            for(size_t j = 0; j < n; ++j) {
                if (ipiv[j]) continue;

                for(size_t k = 0; k < n; ++k) {
                    if (!ipiv[k] && fabs(a[j][k]) > big) {
                        big  = fabs(a[j][k]);
                        irow = j;
                        icol = k;
                    }
                }
            }

            ipiv[icol] = true;

            if (irow != icol)
                for(size_t j = 0; j < n; ++j)
                    std::swap(a[irow][j], a[icol][j]);

            idxr[i] = irow;
            idxc[i] = icol;

            precondition(a[icol][icol], "Singular matrix in gaussj");

            value_type pivinv = 1 / a[icol][icol];
            a[icol][icol] = 1;

            for(size_t i = 0; i < n; ++i)
                a[icol][i] *= pivinv;

            for(size_t k = 0; k < n; ++k) {
                if (k != icol) {
                    value_type dum = a[k][icol];
                    a[k][icol] = 0;
                    for(size_t i = 0; i < n; ++i)
                        a[k][i] -= a[icol][i] * dum;
                }
            }
        }

        for(size_t i = n; i-- > 0; ) {
            if (idxr[i] != idxc[i]) {
                for(size_t j = 0; j < n; ++j)
                    std::swap(a[j][idxr[i]], a[j][idxc[i]]);
            }
        }
    }

} // namespace detail
} // namespace amgcl

#endif
