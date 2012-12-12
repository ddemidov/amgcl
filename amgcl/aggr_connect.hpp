#ifndef AMGCL_AGGR_CONNECT_HPP
#define AMGCL_AGGR_CONNECT_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   aggr_connect.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Strong couplings for aggregation-based AMG.
 */

#include <vector>

namespace amgcl {
namespace aggr {

/// Strong couplings for aggregation-based AMG.
/**
 * \param A The system matrix
 * \param eps_strong ///< copydoc amgcl::interp::aggregation::params::eps_strong
 *
 * \returns vector of bools (actually chars) corresponding to A's nonzero entries.
 */
template <class spmat>
std::vector<char> connect(const spmat &A, float eps_strong) {
    typedef typename sparse::matrix_index<spmat>::type index_t;
    typedef typename sparse::matrix_value<spmat>::type value_t;

    const index_t n = sparse::matrix_rows(A);

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

    std::vector<char> S(sparse::matrix_nonzeros(A));

    auto dia = sparse::diagonal(A);

    value_t eps2 = eps_strong * eps_strong;

#pragma omp parallel for schedule(dynamic, 1024)
    for(index_t i = 0; i < n; ++i) {
        value_t eps_dia_i = eps2 * dia[i];

        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
            index_t c = Acol[j];
            value_t v = Aval[j];

            S[j] = (c != i) && (v * v > eps_dia_i * dia[c]);
        }
    }

    return S;
}

} // namespace aggr
} // namespace amgcl
#endif
