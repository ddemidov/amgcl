#ifndef AMGCL_AGGR_PLAIN_HPP
#define AMGCL_AGGR_PLAIN_HPP

#include <vector>
#include <tuple>
#include <algorithm>

#include <amgcl/spmat.hpp>
#include <amgcl/params.hpp>
#include <amgcl/profiler.hpp>

namespace amg {
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
    index_t max_nonzeros_per_row = 0;
    for(index_t i = 0; i < n; ++i) {
        auto w = Arow[i + 1] - Arow[i];
        agg[i] = (w > 1 ? undefined : removed);

        if (w > max_nonzeros_per_row) max_nonzeros_per_row = w;
    }

    std::vector<index_t> neib;
    neib.reserve(max_nonzeros_per_row);

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
            if (c != i) {
                agg[c] = last_g;
                neib.push_back(c);
            }
        }

        // Temporarily mark undefined points adjacent to the new aggregate as
        // beloning to the aggregate. If nobody claims them later, they will
        // stay here.
        for(auto nb = neib.begin(); nb != neib.end(); ++nb) {
            for(index_t j = Arow[*nb], e = Arow[*nb + 1]; j < e; ++j) {
                if (agg[Acol[j]] == undefined) agg[Acol[j]] = last_g;
            }
        }
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

    index_t nc = *std::max_element(aggr.begin(), aggr.end()) + 1;

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

struct aggregated_operator {

    template <class spmat>
    static spmat apply(const spmat &R, const spmat &A, const spmat &P) {
        typedef typename sparse::matrix_index<spmat>::type index_t;
        typedef typename sparse::matrix_value<spmat>::type value_t;

        const auto n  = sparse::matrix_rows(A);
        const auto nc = sparse::matrix_rows(R);

        spmat a(nc, nc);

        std::fill(a.row.begin(), a.row.end(), static_cast<index_t>(0));

        std::vector<index_t> marker(nc, static_cast<index_t>(-1));

        for(index_t i = 0; i < n; ++i) {
            assert(P.row[i] == P.row[i + 1] || P.row[i] + 1 == P.row[i + 1]);
            if (P.row[i] == P.row[i + 1]) continue;

            index_t gi = P.col[P.row[i]];

            assert(gi < nc && "Wrong aggregation data");

            for(index_t j = A.row[i], e = A.row[i+1]; j < e; ++j) {
                index_t c = A.col[j];

                if (P.row[c] == P.row[c + 1]) continue;

                index_t gj = P.col[P.row[c]];

                assert(gj < nc && "Wrong aggregation data");

                if (marker[gj] != gi) {
                    marker[gj] = gi;
                    ++( a.row[gi + 1] );
                }
            }
        }

        std::fill(marker.begin(), marker.end(), static_cast<index_t>(-1));

        std::partial_sum(a.row.begin(), a.row.end(), a.row.begin());

        a.reserve(a.row.back());

        for(index_t i = 0; i < n; ++i) {
            if (P.row[i] == P.row[i + 1]) continue;

            index_t gi = P.col[P.row[i]];

            assert(gi < nc && "Wrong aggregation data");

            index_t row_beg  = a.row[gi];
            index_t row_end  = a.row[gi + 1];
            index_t row_head = row_beg;

            for(index_t j = A.row[i], e = A.row[i+1]; j < e; ++j) {
                index_t c = A.col[j];

                if (P.row[c] == P.row[c + 1]) continue;

                index_t gj = P.col[P.row[c]];
                index_t v  = A.val[j];

                assert(gj < nc && "Wrong aggregation data");

                if (marker[gj] < row_beg || marker[gj] >= row_end) {
                    marker[gj] = row_head;
                    a.col[row_head] = gj;
                    a.val[row_head] = v;
                    ++row_head;
                } else {
                    a.val[marker[gj]] += v;
                }
            }
        }

        return a;
    }

};

/*
template <class T>
struct coarse_operator< aggregation<T> > {
    typedef aggregated_operator type;
};
*/

} // namespace interp
} // namespace amg

#endif
