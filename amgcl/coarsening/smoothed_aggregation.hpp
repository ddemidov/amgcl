#ifndef AMGCL_COARSENING_SMOOTHED_AGGREGATION_HPP
#define AMGCL_COARSENING_SMOOTHED_AGGREGATION_HPP

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
 * \file   amgcl/coarsening/smoothed_aggregation.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Smoothed aggregation coarsening scheme.
 */

#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/galerkin.hpp>
#include <amgcl/tictoc.hpp>

namespace amgcl {
namespace coarsening {

template <class Aggregates>
struct smoothed_aggregation {
    struct params {
        float relax;
        float eps_strong;

        params() : relax(0.666f), eps_strong(0.08f) {}
    };

    template <typename Val, typename Col, typename Ptr>
    static boost::tuple<
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >,
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >
        >
    transfer_operators(const backend::crs<Val, Col, Ptr> &A, params &prm)
    {
        typedef backend::crs<Val, Col, Ptr> matrix;

        const size_t n = rows(A);

        TIC("aggregates");
        Aggregates aggr(A, prm.eps_strong);
        prm.eps_strong *= 0.5;
        TOC("aggregates");

        TIC("interpolation");
        boost::shared_ptr<matrix> P = boost::make_shared<matrix>();
        P->nrows = n;
        P->ncols = aggr.count;
        P->ptr.resize(n + 1);

        boost::fill(P->ptr, 0);

#pragma omp parallel
        {
            std::vector<long> marker(aggr.count, -1);

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            size_t chunk_size  = (n + nt - 1) / nt;
            size_t chunk_start = tid * chunk_size;
            size_t chunk_end   = std::min(n, chunk_start + chunk_size);
#else
            size_t chunk_start = 0;
            size_t chunk_end   = n;
#endif

            // Count number of entries in P.
            for(size_t i = chunk_start; i < chunk_end; ++i) {
                for(Ptr j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                    size_t c = static_cast<size_t>(A.col[j]);

                    // Skip weak off-diagonal connections.
                    if (c != i && !aggr.strong_connection[j])
                        continue;

                    long g = aggr.id[c];

                    if (g >= 0 && static_cast<size_t>(marker[g]) != i) {
                        marker[g] = i;
                        ++( P->ptr[i + 1] );
                    }
                }
            }

            boost::fill(marker, -1);

#pragma omp barrier
#pragma omp single
            {
                boost::partial_sum(P->ptr, P->ptr.begin());
                P->col.resize(P->ptr.back());
                P->val.resize(P->ptr.back());
            }

            // Fill the interpolation matrix.
            for(size_t i = chunk_start; i < chunk_end; ++i) {

                // Diagonal of the filtered matrix is the original matrix
                // diagonal minus its weak connections.
                Val dia = 0;
                for(Ptr j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                    if (static_cast<size_t>(A.col[j]) == i)
                        dia += A.val[j];
                    else if (!aggr.strong_connection[j])
                        dia -= A.val[j];
                }
                dia = 1 / dia;

                Ptr row_beg = P->ptr[i];
                Ptr row_end = row_beg;
                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
                    size_t c = static_cast<size_t>(A.col[j]);

                    // Skip weak couplings, ...
                    if (c != i && !aggr.strong_connection[j]) continue;

                    // ... and the ones not in any aggregate.
                    long g = aggr.id[c];
                    if (g < 0) continue;

                    Val v = (c == i) ? 1 - prm.relax : -prm.relax * dia * A.val[j];

                    if (marker[g] < row_beg) {
                        marker[g] = row_end;
                        P->col[row_end] = g;
                        P->val[row_end] = v;
                        ++row_end;
                    } else {
                        P->val[ marker[g] ] += v;
                    }
                }
            }
        }
        TOC("interpolation");

        boost::shared_ptr<matrix> R = boost::make_shared<matrix>();
        *R = transpose(*P);

        return boost::make_tuple(P, R);
    }

    template <typename Val, typename Col, typename Ptr>
    static boost::shared_ptr< backend::crs<Val, Col, Ptr> >
    coarse_operator(
            const backend::crs<Val, Col, Ptr> &A,
            const backend::crs<Val, Col, Ptr> &P,
            const backend::crs<Val, Col, Ptr> &R,
            const params&
            )
    {
        return detail::galerkin(A, P, R);
    }
};

} // namespace coarsening
} // namespace amgcl

#endif
