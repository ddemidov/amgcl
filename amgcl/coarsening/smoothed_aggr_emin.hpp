#ifndef AMGCL_COARSENING_SMOOTHED_AGGR_EMIN_HPP
#define AMGCL_COARSENING_SMOOTHED_AGGR_EMIN_HPP

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
 * \file   amgcl/coarsening/smoothed_aggr_emin.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Smoothed aggregation with energy minimization coarsening.
 */

#include <limits>

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/galerkin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/detail/sort_row.hpp>

namespace amgcl {
namespace coarsening {

/// Smoothed aggregation with energy minimization.
/**
 * \param Aggregates \ref aggregates formation.
 * \ingroup coarsening
 * \sa \cite Sala2008
 */
template <class Aggregates>
struct smoothed_aggr_emin {
    /// Coarsening parameters.
    struct params {
        /// Aggregation parameters.
        typename Aggregates::params aggr;

        params() {}

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_CHILD(p, aggr)
        {}
    };

    /// \copydoc amgcl::coarsening::aggregation::transfer_operators
    template <class Matrix>
    static boost::tuple<
        boost::shared_ptr<Matrix>,
        boost::shared_ptr<Matrix>
        >
    transfer_operators(const Matrix &A, params &prm)
    {
        typedef typename backend::value_type<Matrix>::type Val;
        const size_t n = rows(A);

        TIC("aggregates");
        Aggregates aggr(A, prm.aggr);
        prm.aggr.eps_strong *= 0.5;
        TOC("aggregates");

        TIC("interpolation");
        std::vector<Val> D(n);
        std::vector<Val> omega(n);

        boost::shared_ptr<Matrix> P = interpolation(A, aggr, D, omega);
        boost::shared_ptr<Matrix> R = restriction  (A, aggr, D, omega);
        TOC("interpolation");

        return boost::make_tuple(P, R);
    }

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

    private:
        template <typename Val, typename Col, typename Ptr>
        static boost::shared_ptr< backend::crs<Val, Col, Ptr> >
        interpolation(
                const backend::crs<Val, Col, Ptr> &A, const Aggregates &aggr,
                std::vector<Val> &D, std::vector<Val> &omega
                )
        {
            typedef backend::crs<Val, Col, Ptr> matrix;
            const size_t n  = rows(A);
            const size_t nc = aggr.count;

            boost::shared_ptr<matrix> P = boost::make_shared<matrix>();
            P->nrows = n;
            P->ncols = nc;
            P->ptr.resize(n + 1, 0);


            std::vector<Val> omega_p(nc, 0);
            std::vector<Val> denum(nc, 0);

#pragma omp parallel
            {
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

                std::vector<ptrdiff_t> marker(nc, -1);

                // Compute A * P_tent product. P_tent is stored implicitly in aggr.

                // 1. Compute structure of the product result.
                // 2. Store diagonal of filtered matrix.
                for(size_t i = chunk_start; i < chunk_end; ++i) {
                    Val dia = 0;

                    for(Ptr j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                        Col c = A.col[j];
                        Val v = A.val[j];

                        if (static_cast<size_t>(c) == i)
                            dia += v;
                        else if (!aggr.strong_connection[j])
                            dia -= v;

                        if (static_cast<size_t>(c) != i && !aggr.strong_connection[j])
                            continue;

                        ptrdiff_t g = aggr.id[c]; if (g < 0) continue;

                        if (static_cast<size_t>(marker[g]) != i) {
                            marker[g] = i;
                            ++( P->ptr[i + 1] );
                        }
                    }

                    D[i] = dia;
                }

                boost::fill(marker, -1);

#pragma omp barrier
#pragma omp single
                {
                    boost::partial_sum(P->ptr, P->ptr.begin());
                    P->col.resize(P->ptr.back());
                    P->val.resize(P->ptr.back());
                }

                // 2. Compute the product result.
                for(size_t i = chunk_start; i < chunk_end; ++i) {
                    Ptr row_beg = P->ptr[i];
                    Ptr row_end = row_beg;

                    for(Ptr j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                        Col c = A.col[j];

                        if (static_cast<size_t>(c) != i && !aggr.strong_connection[j])
                            continue;

                        ptrdiff_t g = aggr.id[c]; if (g < 0) continue;

                        Val v = (static_cast<size_t>(c) == i ? D[i] : A.val[j]);

                        if (marker[g] < row_beg) {
                            marker[g] = row_end;
                            P->col[row_end] = g;
                            P->val[row_end] = v;
                            ++row_end;
                        } else {
                            P->val[marker[g]] += v;
                        }
                    }

                    // Sort the new row by columns.
                    amgcl::detail::sort_row(
                            &P->col[row_beg],
                            &P->val[row_beg],
                            row_end - row_beg
                            );
                }

                boost::fill(marker, -1);
                std::vector<Col> adap_col(128);
                std::vector<Val> adap_val(128);

#pragma omp barrier

                // Compute A * Dinv * AP row by row and compute columnwise scalar products
                // necessary for computation of omega_p. The actual results of
                // matrix-matrix product are not stored.
                for(size_t ia = chunk_start; ia < chunk_end; ++ia) {
                    adap_col.clear();
                    adap_val.clear();

                    // Form current row of ADAP matrix.
                    for(Ptr ja = A.ptr[ia], ea = A.ptr[ia + 1]; ja < ea; ++ja) {
                        Col ca = A.col[ja];

                        if (static_cast<size_t>(ca) != ia && !aggr.strong_connection[ja])
                            continue;

                        Val dia = D[ca];
                        Val va  = (static_cast<size_t>(ca) == ia ? dia : A.val[ja]);

                        for(Ptr jb = P->ptr[ca], eb = P->ptr[ca + 1]; jb < eb; ++jb) {
                            Col cb = P->col[jb];
                            Val vb = P->val[jb] / dia;

                            if (marker[cb] < 0) {
                                marker[cb] = adap_col.size();
                                adap_col.push_back(cb);
                                adap_val.push_back(va * vb);
                            } else {
                                adap_val[marker[cb]] += va * vb;
                            }
                        }
                    }

                    amgcl::detail::sort_row(
                            adap_col.data(), adap_val.data(), adap_col.size()
                            );

                    // Update columnwise scalar products (AP,ADAP) and (ADAP,ADAP).
                    // 1. (AP, ADAP)
                    for(
                            Ptr ja = P->ptr[ia], ea = P->ptr[ia + 1],
                            jb = 0, eb = adap_col.size();
                            ja < ea && jb < eb;
                       )
                    {
                        Col ca = P->col[ja];
                        Col cb = adap_col[jb];

                        if (ca < cb)
                            ++ja;
                        else if (cb < ca)
                            ++jb;
                        else /*ca == cb*/ {
#pragma omp atomic
                            omega_p[ca] += P->val[ja] * adap_val[jb];
                            ++ja;
                            ++jb;
                        }
                    }

                    // 2. (ADAP, ADAP) (and clear marker)
                    for(size_t j = 0, e = adap_col.size(); j < e; ++j) {
                        Col c = adap_col[j];
                        Val v = adap_val[j];
#pragma omp atomic
                        denum[c] += v * v;
                        marker[c] = -1;
                    }
                }
            }

            boost::transform(omega_p, denum, omega_p.begin(), std::divides<Val>());

            // Convert omega from (4.13) to (4.14) \cite Sala2008:
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                Val w = -1;

                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
                    Col c = A.col[j];
                    if (c != i && !aggr.strong_connection[j])
                        continue;

                    ptrdiff_t g = aggr.id[c]; if (g < 0) continue;
                    if (omega_p[g] < w || w < 0) w = omega_p[g];
                }

                omega[i] = std::max(w, static_cast<Val>(0));
            }

            // Update AP to obtain P.
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                Val wd = omega[i] / D[i];

                for(Ptr j = P->ptr[i], e = P->ptr[i + 1]; j < e; ++j)
                    P->val[j] = (P->col[j] == aggr.id[i] ? 1 : 0) - wd * P->val[j];
            }

            return P;
        }

        template <typename Val, typename Col, typename Ptr>
        static boost::shared_ptr< backend::crs<Val, Col, Ptr> >
        restriction(
                const backend::crs<Val, Col, Ptr> &A, const Aggregates &aggr,
                const std::vector<Val> &D, const std::vector<Val> &omega
                )
        {
            typedef backend::crs<Val, Col, Ptr> matrix;
            const size_t n  = rows(A);
            const size_t nc = aggr.count;

            // Get structure of R_tent from aggr
            std::vector<Ptr> R_tent_ptr(nc + 1, 0);
            for(size_t i = 0; i < n; ++i) {
                ptrdiff_t g = aggr.id[i]; if (g < 0) continue;
                ++R_tent_ptr[g + 1];
            }

            boost::partial_sum(R_tent_ptr, R_tent_ptr.begin());
            std::vector<Col> R_tent_col(R_tent_ptr.back());

            for(size_t i = 0; i < n; ++i) {
                ptrdiff_t g = aggr.id[i]; if (g < 0) continue;
                R_tent_col[R_tent_ptr[g]++] = i;
            }

            std::rotate(R_tent_ptr.begin(), R_tent_ptr.end() - 1, R_tent_ptr.end());
            R_tent_ptr[0] = 0;

            boost::shared_ptr<matrix> R = boost::make_shared<matrix>();
            R->nrows = nc;
            R->ncols = n;
            R->ptr.resize(nc + 1, 0);

            // Compute R_tent * A / D.
#pragma omp parallel
            {
#ifdef _OPENMP
                int nt  = omp_get_num_threads();
                int tid = omp_get_thread_num();

                size_t chunk_size  = (nc + nt - 1) / nt;
                size_t chunk_start = tid * chunk_size;
                size_t chunk_end   = std::min(nc, chunk_start + chunk_size);
#else
                size_t chunk_start = 0;
                size_t chunk_end   = nc;
#endif

                std::vector<ptrdiff_t> marker(n, -1);

                for(size_t ir = chunk_start; ir < chunk_end; ++ir) {
                    for(Ptr jr = R_tent_ptr[ir], er = R_tent_ptr[ir + 1]; jr < er; ++jr) {
                        Col cr = R_tent_col[jr];
                        for(Ptr ja = A.ptr[cr], ea = A.ptr[cr + 1]; ja < ea; ++ja) {
                            Col ca = A.col[ja];
                            if (ca != cr && !aggr.strong_connection[ja]) continue;

                            if (static_cast<size_t>(marker[ca]) != ir) {
                                marker[ca] = ir;
                                ++R->ptr[ir + 1];
                            }
                        }
                    }
                }

                boost::fill(marker, -1);

#pragma omp barrier
#pragma omp single
                {
                    boost::partial_sum(R->ptr, R->ptr.begin());
                    R->col.resize(R->ptr.back());
                    R->val.resize(R->ptr.back());
                }

                for(size_t ir = chunk_start; ir < chunk_end; ++ir) {
                    Ptr row_beg = R->ptr[ir];
                    Ptr row_end = row_beg;

                    for(Ptr jr = R_tent_ptr[ir], er = R_tent_ptr[ir + 1]; jr < er; ++jr) {
                        Col cr = R_tent_col[jr];

                        for(Ptr ja = A.ptr[cr], ea = A.ptr[cr + 1]; ja < ea; ++ja) {
                            Col ca = A.col[ja];
                            if (ca != cr && !aggr.strong_connection[ja]) continue;
                            Val va = (ca == cr ? 1 : (A.val[ja] / D[ca]));

                            if (marker[ca] < row_beg) {
                                marker[ca] = row_end;
                                R->col[row_end] = ca;
                                R->val[row_end] = va;
                                ++row_end;
                            } else {
                                R->val[marker[ca]] += va;
                            }
                        }
                    }
                }
            }

            // Update R.
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(nc); ++i) {
                for(Ptr j = R->ptr[i], e = R->ptr[i + 1]; j < e; ++j) {
                    Col c = R->col[j];
                    R->val[j] = (aggr.id[c] == i ? 1 : 0) - omega[c] * R->val[j];
                }
            }

            return R;
        }
};

} // namespace coarsening
} // namespace amgcl



#endif
