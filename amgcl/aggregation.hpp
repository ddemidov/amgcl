#ifndef AMGCL_AGGR_PLAIN_HPP
#define AMGCL_AGGR_PLAIN_HPP

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
 * \file   aggregation.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Aggregates-based interpolation scheme.
 */

#include <vector>
#include <tuple>
#include <algorithm>

#include <amgcl/spmat.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {

/// Aggreagtion related types and functions.
namespace aggr {

/// Plain aggregation.
/**
 * Modification of a greedy aggregation scheme from (Vanek 1995). Any
 * nonzero matrix entry forms a connection. Variables without neighbours
 * (resulting, e.g., from Dirichlet conditions) are excluded from aggregation
 * process. The aggregation is completed in a single pass over variables:
 * variables adjacent to a new aggregate are temporarily marked as beloning to
 * this aggregate. Later they may be claimed by other aggregates; if nobody
 * claims them, then they just stay in their initial aggregate.
 */
struct plain {

/// Constructs aggregates of variables.
/** 
 * Each entry of the return vector corresponds to a variable and contains
 * number of an aggregate the variable belongs to. If an entry is negative,
 * then variable does not belong to any aggregate.
 *
 * \param A system matrix.
 *
 * \returns a vector of aggregate numbers.
 */
template <class spmat>
static std::vector< typename sparse::matrix_index<spmat>::type >
aggregates( const spmat &A ) {
    typedef typename sparse::matrix_index<spmat>::type index_t;
    typedef typename sparse::matrix_value<spmat>::type value_t;
    
    const index_t n = sparse::matrix_rows(A);

    const index_t undefined = static_cast<index_t>(-1);
    const index_t removed   = static_cast<index_t>(-2);

    std::vector<index_t> agg(n);

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

    // Remove nodes without neighbours
    index_t max_row_width = 0;
    for(index_t i = 0; i < n; ++i) {
        auto w = Arow[i + 1] - Arow[i];
        agg[i] = (w > 1 ? undefined : removed);

        if (w > max_row_width) max_row_width = w;
    }

    std::vector<index_t> neib;
    neib.reserve(max_row_width);

    index_t last_g = static_cast<index_t>(-1);

    // Perform plain aggregation
    for(index_t i = 0; i < n; ++i) {
        if (agg[i] != undefined) continue;

        // The point is not adjacent to a core of any previous aggregate:
        // so its a seed of a new aggregate.
        agg[i] = ++last_g;

        neib.clear();

        // Include its neighbors as well.
        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
            index_t c = Acol[j];
            if (c != i && agg[c] != removed) {
                agg[c] = last_g;
                neib.push_back(c);
            }
        }

        // Temporarily mark undefined points adjacent to the new aggregate as
        // beloning to the aggregate. If nobody claims them later, they will
        // stay here.
        for(auto nb = neib.begin(); nb != neib.end(); ++nb)
            for(index_t j = Arow[*nb], e = Arow[*nb + 1]; j < e; ++j)
                if (agg[Acol[j]] == undefined) agg[Acol[j]] = last_g;
    }

    assert( std::count(agg.begin(), agg.end(), undefined) == 0 );

    return agg;
}

};

} // namespace aggr

namespace interp {

/// Aggregation-based interpolation scheme.
/**
 * \param aggr_type Aggregation scheme. For now the only possible value is
 *                  amgcl::aggr::plain.
 */
template <class aggr_type>
struct aggregation {

/// Parameters controlling aggregation.
struct params {
    float over_interp;   ///< Over-interpolation factor.

    params() : over_interp(1.5) {}
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

    P.col.reserve(n);
    P.val.reserve(n);

    P.row[0] = 0;
    for(index_t i = 0; i < n; ++i) {
        if (aggr[i] >= 0) {
            P.row[i + 1] = P.row[i] + 1;
            P.col.push_back(aggr[i]);
            P.val.push_back(static_cast<value_t>(1));
        } else {
            P.row[i + 1] = P.row[i];
        }
    }
    TOC("interpolation");

    return P;
}

};

/// Coarse level computing for aggregation-based AMG.
struct aggregated_operator {
    template <class spmat, class Params>
    static spmat apply(const spmat &R, const spmat &A, const spmat &P,
            const Params &prm)
    {
        typedef typename sparse::matrix_value<spmat>::type value_t;

        // For now this s just a Galerking operator with possible
        // over-interpolation.
        auto a = sparse::prod(sparse::prod(R, A), P);

        if (prm.over_interp > 1.0f)
            std::transform(a.val.begin(), a.val.end(), a.val.begin(),
                    [&prm](value_t v) {
                        return v / prm.over_interp;
                    });
    }
};

template <class T>
struct coarse_operator< aggregation<T> > {
    typedef aggregated_operator type;
};

} // namespace interp
} // namespace amgcl

#endif
