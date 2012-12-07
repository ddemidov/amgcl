#ifndef AMGCL_AGGR_PLAIN_HPP
#define AMGCL_AGGR_PLAIN_HPP

#include <vector>
#include <tuple>
#include <algorithm>

#include <amgcl/spmat.hpp>
#include <amgcl/params.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
namespace aggr {

struct plain {

template <class spmat>
static std::vector< typename sparse::matrix_index<spmat>::type >
aggregates( const spmat &A, const params &prm ) {
    typedef typename sparse::matrix_index<spmat>::type index_t;
    typedef typename sparse::matrix_value<spmat>::type value_t;
    
    const index_t n = sparse::matrix_rows(A);

    const index_t undefined = static_cast<index_t>(-1);
    const index_t removed   = static_cast<index_t>(-2);

    std::vector<index_t> agg(n);

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

    // Remove nodes without neigbors
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

// Constructs corse level by agregation.
template <class aggr_type>
struct aggregation {

template < class value_t, class index_t >
static sparse::matrix<value_t, index_t> interp(
        const sparse::matrix<value_t, index_t> &A, const params &prm
        )
{
    const index_t n = sparse::matrix_rows(A);

    TIC("aggregates");
    auto aggr = aggr_type::aggregates(A, prm);
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

} // namespace interp
} // namespace amgcl

#endif
