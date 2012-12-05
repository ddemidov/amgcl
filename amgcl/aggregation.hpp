#ifndef AMGCL_AGGR_PLAIN_HPP
#define AMGCL_AGGR_PLAIN_HPP

#include <vector>
#include <tuple>
#include <algorithm>

#include <amgcl/spmat.hpp>
#include <amgcl/params.hpp>

namespace amg {

#ifdef AMGCL_PROFILING
#  include <amgcl/profiler.hpp>
#  define TIC(what) prof.tic(what);
#  define TOC(what) prof.tic(what);
   extern amg::profiler<> prof;
#else
#  define TIC(what)
#  define TOC(what)
#endif

namespace aggr {

struct plain {

template <class spmat>
static std::vector< typename sparse::matrix_index<spmat>::type >
aggregates( const spmat &A, const params &prm ) {
    typedef typename sparse::matrix_index<spmat>::type index_t;
    typedef typename sparse::matrix_value<spmat>::type value_t;
    
    const index_t n = sparse::matrix_rows(A);

    index_t n_undef = n;
    index_t last_g  = -1;

    std::vector<char>    def(n, false);
    std::vector<index_t> agg(n, -1);
    std::vector<value_t> dia(diagonal(A));

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

    // Compute strongly coupled neighborhoods.
    std::vector<index_t> Srow;
    Srow.reserve(n + 1);
    Srow.push_back(0);

    std::vector<index_t> Scol;
    Scol.reserve( sparse::matrix_nonzeros(A) );

    for(index_t i = 0; i < n; ++i) {
        value_t dia_i;

        assert(dia_i != 0 && "Zero diagonal is bad anyway");

        index_t j = Arow[i];
        index_t e = Arow[i + 1];

        if (j + 1 == e) {
            def[i] = true;
            --n_undef;
        }

        for( ; j < e; ++j) {
            index_t c = Acol[j];
            value_t v = Aval[j];

            if (v * v > prm.eps_strong * dia_i * dia[c])
                Scol.push_back(c);
        }

        Srow.push_back(Scol.size());
    }

    for(index_t i = 0; i < n; ++i) {
        if (def[i]) continue;

        ++last_g;

        for(index_t j = Srow[i], e = Srow[i + 1]; j < e; ++j) {
            index_t c = Scol[j];

            if (def[c]) continue;

            def[c] = true;
            agg[c] = last_g;

            --n_undef;
        }
    }

    assert(n_undef == 0);

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

} // namespace interp
} // namespace amg

#endif
