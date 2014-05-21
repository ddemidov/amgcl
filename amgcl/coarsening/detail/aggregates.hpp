#ifndef AMGCL_COARSENING_DETAIL_AGGREGATES_HPP
#define AMGCL_COARSENING_DETAIL_AGGREGATES_HPP

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
 * \file   amgcl/coarsening/detail/aggregates.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Plain aggregation.
 */

#include <vector>
#include <boost/foreach.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace coarsening {
namespace detail {

/// Constructs aggregates for the given matrix.
/**
 * Modification of a greedy aggregation scheme from \ref Vanek_1996 "Vanek
 * (1996)".  Any nonzero matrix entry forms a connection. Variables without
 * neighbours (resulting, e.g., from Dirichlet conditions) are excluded from
 * aggregation process. The aggregation is completed in a single pass over
 * variables: variables adjacent to a new aggregate are temporarily marked as
 * beloning to this aggregate. Later they may be claimed by other aggregates;
 * if nobody claims them, then they just stay in their initial aggregate.
 *
 * Each entry of the return vector corresponds to a variable and contains
 * number of an aggregate the variable belongs to. If an entry is negative,
 * then variable does not belong to any aggregate.
 *
 * \param      A The system matrix.
 * \param      S Strong couplings in A.
 * \param[out] G Aggregate id for each row in A.
 *
 * \returns number of aggregates
 */
template <typename Val, typename Col, typename Ptr>
size_t aggregates(
        const backend::crs<Val, Col, Ptr> &A, float eps_strong,
        std::vector<Col> &G
        )
{
    typedef typename backend::crs<Val, Col, Ptr>::row_iterator row_iterator;
    const size_t n = rows(A);

    // Determine strong couplings for the matrix.
    std::vector<char> S( nonzeros(A) );
    Val eps2 = eps_strong * eps_strong;
    std::vector<Val> dia = diagonal(A);

#pragma omp parallel for
    for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        Val eps_dia_i = eps2 * dia[i];

        // Determine connections strength:
        for(Ptr j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
            Col c = A.col[j];
            Val v = A.val[j];

            S[j] = (c != i) && (v * v > eps_dia_i * dia[c]);
        }
    }


    // Proceed with aggregation.
    const Col undefined = static_cast<Col>(-1);
    const Col removed   = static_cast<Col>(-2);

    G.resize(n);

    // Remove nodes without neighbours
    size_t max_neib = 0;
    for(size_t i = 0; i < n; ++i) {
        Ptr j = A.ptr[i], e = A.ptr[i+1];
        max_neib = std::max<size_t>(max_neib, e - j);

        Col state = removed;
        for(; j < e; ++j) if (S[j]) {
            state = undefined;
            break;
        }

        G[i] = state;
    }

    std::vector<Col> neib;
    neib.reserve(max_neib);

    Col gmax = static_cast<Col>(-1);

    // Perform plain aggregation
    for(size_t i = 0; i < n; ++i) {
        if (G[i] != undefined) continue;

        // The point is not adjacent to a core of any previous aggregate:
        // so its a seed of a new aggregate.
        G[i] = ++gmax;

        neib.clear();

        // Include its neighbors as well.
        for(Ptr j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
            Col c = A.col[j];
            if (S[j] && G[c] != removed) {
                G[c] = gmax;
                neib.push_back(c);
            }
        }

        // Temporarily mark undefined points adjacent to the new aggregate as
        // beloning to the aggregate. If nobody claims them later, they will
        // stay here.
        BOOST_FOREACH(Col c, neib)
            for(Ptr j = A.ptr[c], e = A.ptr[c+1]; j < e; ++j)
                if (S[j] && G[A.col[j]] == undefined) G[A.col[j]] = gmax;
    }

    return gmax + 1;
}


} // namespace detail
} // namespace coarsening
} // namespace amgcl

#endif
