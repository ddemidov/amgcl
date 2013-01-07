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
#include <cassert>

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

    BOOST_AUTO(Arow, sparse::matrix_outer_index(A));
    BOOST_AUTO(Acol, sparse::matrix_inner_index(A));
    BOOST_AUTO(Aval, sparse::matrix_values(A));

    std::vector<char> S(sparse::matrix_nonzeros(A));

    BOOST_AUTO(dia, sparse::diagonal(A));

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

/// Builds reduced matrix for non-scalar system of equations.
template <class spmat>
sparse::matrix<
    typename sparse::matrix_value<spmat>::type,
    typename sparse::matrix_index<spmat>::type
    >
pointwise_matrix(const spmat &A, unsigned dof_per_node) {
    typedef typename sparse::matrix_value<spmat>::type value_t;
    typedef typename sparse::matrix_index<spmat>::type index_t;

    const index_t n = sparse::matrix_rows(A);
    const index_t m = sparse::matrix_cols(A);

    assert(n % dof_per_node == 0
            && "Number of rows in matrix should be divisible by dof_per_node");
    assert(m % dof_per_node == 0
            && "Number of cols in matrix should be divisible by dof_per_node");

    const index_t nc = n / dof_per_node;
    const index_t mc = m / dof_per_node;

    BOOST_AUTO(Arow, sparse::matrix_outer_index(A));
    BOOST_AUTO(Acol, sparse::matrix_inner_index(A));
    BOOST_AUTO(Aval, sparse::matrix_values(A));

    sparse::matrix<value_t, index_t> B(nc, mc);
    std::fill(B.row.begin(), B.row.end(), static_cast<index_t>(0));

    // Count number of nonzeros in block matrix.
    std::vector<index_t> marker(mc, static_cast<index_t>(-1));
    for(index_t ib = 0; ib < nc; ++ib) {
        for(index_t ia = ib * dof_per_node, ei = ia + dof_per_node; ia < ei; ++ia) {
            for(index_t ja = Arow[ia], ej = Arow[ia + 1]; ja < ej; ++ja) {
                index_t cb = Acol[ja] / dof_per_node;
                if (marker[cb] != ib) {
                    marker[cb] = ib;
                    ++B.row[ib + 1];
                }
            }
        }
    }
    std::partial_sum(B.row.begin(), B.row.end(), B.row.begin());
    std::fill(marker.begin(), marker.end(), static_cast<index_t>(-1));

    // Fill the reduced matrix. Use Frobenius norm for blocks.
    B.reserve(B.row.back());

    for(index_t ib = 0; ib < nc; ++ib) {
        index_t row_beg = B.row[ib];
        index_t row_end = row_beg;

        for(index_t ia = ib * dof_per_node, ei = ia + dof_per_node; ia < ei; ++ia) {
            for(index_t ja = Arow[ia], ej = Arow[ia + 1]; ja < ej; ++ja) {
                index_t cb = Acol[ja] / dof_per_node;
                value_t va = Aval[ja];

                if (marker[cb] < row_beg) {
                    marker[cb] = row_end;
                    B.col[row_end] = cb;
                    B.val[row_end] = va * va;
                    ++row_end;
                } else {
                    B.val[marker[cb]] += va * va;
                }
            }
        }

        for(index_t k = row_beg; k < row_end; ++k)
            B.val[k] = sqrt(B.val[k]);
    }

    return B;
}

} // namespace aggr
} // namespace amgcl
#endif
