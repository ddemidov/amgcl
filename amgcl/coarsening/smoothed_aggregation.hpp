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

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/galerkin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace coarsening {

/// Smoothed aggregation coarsening.
/**
 * \param Aggregates \ref aggregates formation.
 * \ingroup coarsening
 * \sa \cite Vanek1996
 */
template <class Aggregates>
struct smoothed_aggregation {
    /// Coarsening parameters
    struct params {
        /// Aggregation parameters.
        typename Aggregates::params aggr;

        /// Relaxation factor \f$\omega\f$.
        /**
         * Piecewise constant prolongation \f$\tilde P\f$ from non-smoothed
         * aggregation is improved by a smoothing to get the final prolongation
         * matrix \f$P\f$. Simple Jacobi smoother is used here, giving the
         * prolongation matrix
         * \f[P = \left( I - \omega D^{-1} A^F \right) \tilde P.\f]
         * Here \f$A^F = (a_{ij}^F)\f$ is the filtered matrix given by
         * \f[
         * a_{ij}^F =
         * \begin{cases}
         * a_{ij} \quad \text{if} \; j \in N_i\\
         * 0 \quad \text{otherwise}
         * \end{cases}, \quad \text{if}\; i \neq j,
         * \quad a_{ii}^F = a_{ii} - \sum\limits_{j=1,j\neq i}^n
         * \left(a_{ij} - a_{ij}^F \right),
         * \f]
         * where \f$N_i\f$ is the set of variables, strongly coupled to
         * variable \f$i\f$, and \f$D\f$ denotes the diagonal of \f$A^F\f$.
         */
        float relax;

        params() : relax(0.666f) {
            aggr.eps_strong = 0.08f;
        }
    };

    /// \copydoc amgcl::coarsening::aggregation::transfer_operators
    template <class Matrix>
    static boost::tuple< boost::shared_ptr<Matrix>, boost::shared_ptr<Matrix> >
    transfer_operators(const Matrix &A, params &prm)
    {
        typedef typename backend::value_type<Matrix>::type Val;

        const size_t n = rows(A);

        TIC("aggregates");
        Aggregates aggr(A, prm.aggr);
        prm.aggr.eps_strong *= 0.5;
        TOC("aggregates");

        TIC("interpolation");
        boost::shared_ptr<Matrix> P = boost::make_shared<Matrix>();
        P->nrows = n;
        P->ncols = aggr.count;
        P->ptr.resize(n + 1, 0);

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
                for(long j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                    size_t c = static_cast<size_t>(A.col[j]);

                    // Skip weak off-diagonal connections.
                    if (c != i && !aggr.strong_connection[j])
                        continue;

                    long g = aggr.id[c];

                    if (g >= 0 && static_cast<size_t>(marker[g]) != i) {
                        marker[g] = static_cast<long>(i);
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
                for(long j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                    if (static_cast<size_t>(A.col[j]) == i)
                        dia += A.val[j];
                    else if (!aggr.strong_connection[j])
                        dia -= A.val[j];
                }
                dia = 1 / dia;

                long row_beg = P->ptr[i];
                long row_end = row_beg;
                for(long j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
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

        boost::shared_ptr<Matrix> R = boost::make_shared<Matrix>();
        *R = transpose(*P);

        return boost::make_tuple(P, R);
    }

    /// \copydoc amgcl::coarsening::aggregation::coarse_operator
    template <class Matrix>
    static boost::shared_ptr<Matrix>
    coarse_operator(
            const Matrix &A,
            const Matrix &P,
            const Matrix &R,
            const params&
            )
    {
        return detail::galerkin(A, P, R);
    }
};

} // namespace coarsening
} // namespace amgcl

#endif
