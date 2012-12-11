#ifndef AMGCL_INTERP_SMOOTHED_AGGR_HPP
#define AMGCL_INTERP_SMOOTHED_AGGR_HPP

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
 * \file   interp_smoothed_aggr.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Smoothed aggregates-based interpolation scheme.
 */

#include <vector>
#include <algorithm>

#include <amgcl/spmat.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {

namespace interp {

/// Smoothed aggregation-based interpolation scheme.
/**
 * See \ref Vanek_1996 "Vanek (1996)"
 *
 * \param aggr_type \ref aggregation "Aggregation scheme".
 *
 * \ingroup interpolation
 */
template <class aggr_type>
struct smoothed_aggregation {

/// Parameters controlling aggregation.
struct params {
    /// Relaxation factor \f$\omega\f$.
    /**
     * See \ref Vanek_1996 "Vanek (1996)".
     * Piecewise constant prolongation \f$\tilde P\f$ from \ref
     * amgcl::interp::aggregation is improved by a smoothing to get the final
     * prolongation matrix \f$P\f$. Simple Jacobi smoother is used here, giving
     * the prolongation matrix
     * \f[P = \left( I - \omega D^{-1} A^F \right) \tilde P.\f]
     * Here \f$A^F = (a_{ij}^F)\f$ is the filtered matrix given by
     * \f[
     * a_{ij}^F =
     * \begin{cases}
     * a_{ij} \quad \text{if} \; j \in N_i(\varepsilon)\\
     * 0 \quad \text{otherwise}
     * \end{cases}, \quad \text{if}\; i \neq j,
     * \quad a_{ii}^F = a_{ii} - \sum\limits_{j=1,j\neq i}^n
     * \left(a_{ij} - a_{ij}^F \right),
     * \f]
     * where \f$D\f$ denotes the diagonal of \f$A^F\f$.
     */
    float relax;

    /// Matrix filtering parameter \f$\varepsilon\f$
    /**
     * \sa relax
     */
    float eps;

    params() : relax(0.666f), eps(0.1f) {}
};

/// Constructs coarse level by agregation.
/**
 * Returns interpolation operator, which is enough to construct system matrix
 * at coarser level.
 *
 * \param A   system matrix.
 * \param prm parameters.
 *
 * \returns interpolation operator.
 */
// TODO: actually filter the matrix.
template < class value_t, class index_t >
static sparse::matrix<value_t, index_t> interp(
        const sparse::matrix<value_t, index_t> &A, const params &prm
        )
{
    const index_t n = sparse::matrix_rows(A);

    TIC("aggregates");
    auto aggr = aggr_type::aggregates(A);
    TOC("aggregates");

    index_t nc = std::max(
            static_cast<index_t>(0),
            *std::max_element(aggr.begin(), aggr.end()) + static_cast<index_t>(1)
            );

    TIC("interpolation");
    sparse::matrix<value_t, index_t> P(n, nc);
    std::fill(P.row.begin(), P.row.end(), static_cast<index_t>(0));

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

    std::vector<index_t> marker(nc, static_cast<index_t>(-1));

    // Count number of entries in P.
    index_t nnz = 0;
    for(index_t i = 0; i < n; ++i) {
        for(index_t j = Arow[i], e = Arow[i+1]; j < e; ++j) {
            index_t g = aggr[Acol[j]];

            if (g >= 0 && marker[g] != i) {
                marker[g] = i;
                ++P.row[i + 1];
            }
        }
    }

    std::fill(marker.begin(), marker.end(), static_cast<index_t>(-1));

    std::partial_sum(P.row.begin(), P.row.end(), P.row.begin());
    P.reserve(P.row.back());

    // Fill the interpolation matrix.
    for(index_t i = 0; i < n; ++i) {
        value_t dia = 0;

        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
            if (Acol[j] == i) {
                dia = Aval[j];
                break;
            }
        }

        dia = 1 / dia;

        index_t row_beg = P.row[i];
        index_t row_end = row_beg;
        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
            index_t c = Acol[j];
            index_t g = aggr[c];

            if (g < 0) continue;

            value_t v = -prm.relax * Aval[j] * dia;
            if (c == i) v += static_cast<value_t>(1);

            if (marker[g] < row_beg) {
                marker[g] = row_end;
                P.col[row_end] = g;
                P.val[row_end] = v;
                ++row_end;
            } else {
                P.val[marker[g]] += v;
            }
        }
    }
    TOC("interpolation");

    return P;
}

};

} // namespace interp
} // namespace amgcl



#endif
