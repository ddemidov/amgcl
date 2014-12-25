#ifndef AMGCL_COARSENING_AGGREGATION_HPP
#define AMGCL_COARSENING_AGGREGATION_HPP

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
 * \file   amgcl/coarsening/aggregation.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Non-smoothed aggregation coarsening.
 */

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/algorithm.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/scaled_galerkin.hpp>
#include <amgcl/util.hpp>

#ifdef AMGCL_HAVE_EIGEN
#  include <Eigen/Dense>
#  include <Eigen/QR>
#endif

namespace amgcl {

/// Coarsening strategies
namespace coarsening {

/**
 * \defgroup coarsening Coarsening strategies
 * \brief Coarsening strategies for AMG hirarchy construction.
 *
 * A coarsener in AMGCL is a class that takes a system matrix and returns three
 * operators:
 *
 * 1. Restriction operator R that downsamples the residual error to a
 *    coarser level in AMG hierarchy,
 * 2. Prolongation operator P that interpolates a correction computed on a
 *    coarser grid into a finer grid,
 * 3. System matrix \f$A^H\f$ at a coarser level that is usually computed as a
 *    Galerkin operator \f$A^H = R A^h P\f$.
 *
 * The AMG hierarchy is constructed by recursive invocation of the selected
 * coarsener.
 */

/// Non-smoothed aggregation.
/**
 * \param Aggregates \ref aggregates formation.
 * \ingroup coarsening
 */
template <class Aggregates>
struct aggregation {
    /// Coarsening parameters.
    struct params {
        /// Aggregation parameters.
        typename Aggregates::params aggr;

        /// Over-interpolation factor \f$\alpha\f$.
        /**
         * In case of aggregation coarsening, coarse-grid
         * correction of smooth error, and by this the overall convergence, can
         * often be substantially improved by using "over-interpolation", that is,
         * by multiplying the actual correction (corresponding to piecewise
         * constant interpolation) by some factor \f$\alpha > 1\f$. Equivalently,
         * this means that the coarse-level Galerkin operator is re-scaled by
         * \f$1 / \alpha\f$:
         * \f[I_h^HA_hI_H^h \to \frac{1}{\alpha}I_h^HA_hI_H^h.\f]
         *
         * \sa  \cite Stuben1999, Section 9.1 "Re-scaling of the Galerkin operator".
         */
        float over_interp;

#ifdef AMGCL_HAVE_EIGEN
        /// Number of vectors in problem's null-space.
        int Bcols;

        /// The vectors in problem's null-space.
        /**
         * The 2D matrix B is stored row-wise in a continuous vector. Here row
         * corresponds to a degree of freedom in the original problem, and
         * column corresponds to a vector in the problem's null-space.
         */
        std::vector<double> B;
#endif

        params()
            : over_interp(1.5f)
#ifdef AMGCL_HAVE_EIGEN
            , Bcols(0)
#endif
        { }

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_CHILD(p, aggr)
            , AMGCL_PARAMS_IMPORT_VALUE(p, over_interp)
#ifdef AMGCL_HAVE_EIGEN
            , AMGCL_PARAMS_IMPORT_VALUE(p, Bcols)
#endif
        {
#ifdef AMGCL_HAVE_EIGEN
            double *b = 0;
            size_t Brows = 0;

            b     = p.get("B",     b);
            Brows = p.get("Brows", Brows);

            if (b) {
                precondition(Bcols > 0,
                        "Error in aggregation parameters: "
                        "B is set, but Bcols is not"
                        );

                precondition(Brows > 0,
                        "Error in aggregation parameters: "
                        "B is set, but Brows is not"
                        );

                B.assign(b, b + Brows * Bcols);
            }
#endif
        }
    };

    /// Creates transfer operators for the given system matrix.
    /**
     * \param A   The system matrix.
     * \param prm Coarsening parameters.
     * \returns   A tuple of prolongation and restriction operators.
     */
    template <class Matrix>
    static boost::tuple<
        boost::shared_ptr<Matrix>,
        boost::shared_ptr<Matrix>
        >
    transfer_operators(const Matrix &A, params &prm)
    {
        typedef typename backend::value_type<Matrix>::type V;

        const size_t n = rows(A);

        TIC("aggregates");
        Aggregates aggr(A, prm.aggr);
        TOC("aggregates");

        TIC("interpolation");
        boost::shared_ptr<Matrix> P = boost::make_shared<Matrix>();

#ifdef AMGCL_HAVE_EIGEN
        if (prm.Bcols > 0) {
            precondition(!prm.B.empty(),
                    "Error in aggregation parameters: "
                    "Bcols > 0, but B is empty"
                    );

            // Improve tentative prolongation by using null-space
            std::vector<ptrdiff_t> order(
                    boost::counting_iterator<ptrdiff_t>(0),
                    boost::counting_iterator<ptrdiff_t>(n)
                    );

            boost::sort(order, [&aggr](ptrdiff_t i, ptrdiff_t j){
                    // Cast to unsigned type to keep negative values at the end
                    return
                        static_cast<size_t>(aggr.id[i]) <
                        static_cast<size_t>(aggr.id[j]);
                    });

            P->nrows = n;
            P->ncols = aggr.count * prm.Bcols;
            P->ptr.reserve(n + 1);

            P->ptr.push_back(0);
            for(size_t i = 0; i < n; ++i) {
                if (aggr.id[i] < 0)
                    P->ptr.push_back(P->ptr.back());
                else
                    P->ptr.push_back(P->ptr.back() + prm.Bcols);
            }

            P->col.resize(P->ptr.back());
            P->val.resize(P->ptr.back());

            std::vector<double> Bnew;
            Bnew.reserve(aggr.count * prm.Bcols * prm.Bcols);

            typedef Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> EMatrix;
            typedef Eigen::Map<EMatrix> EMap;

            size_t offset = 0;

            std::vector<double> Bdata;
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(aggr.count); ++i) {
                int d = 0;
                Bdata.clear();
                for(ptrdiff_t j = offset; aggr.id[order[j]] == i; ++j, ++d) {
                    double *Brow = &prm.B[order[j] * prm.Bcols];
                    for(int k = 0; k < prm.Bcols; ++k)
                        Bdata.push_back(Brow[k]);
                }
                EMap Bpart(Bdata.data(), d, prm.Bcols);
                Eigen::HouseholderQR<EMatrix> qr(Bpart);

                EMatrix R = qr.matrixQR().template triangularView<Eigen::Upper>();
                EMatrix Q = qr.householderQ();

                double sign = R(0,0) > 0 ? 1 : -1;
                for(int ii = 0; ii < prm.Bcols; ++ii)
                    for(int jj = 0; jj < prm.Bcols; ++jj)
                        Bnew.push_back( R(ii,jj) * sign );

                for(int ii = 0; ii < d; ++ii, ++offset) {
                    auto c = &P->col[P->ptr[order[offset]]];
                    auto v = &P->val[P->ptr[order[offset]]];

                    for(int jj = 0; jj < prm.Bcols; ++jj) {
                        c[jj] = i * prm.Bcols + jj;
                        v[jj] = Q(ii,jj) * sign;
                    }
                }
            }

            swap(prm.B, Bnew);
        } else {
#endif
            P->nrows = n;
            P->ncols = aggr.count;
            P->ptr.reserve(n + 1);
            P->col.reserve(n);

            P->ptr.push_back(0);
            for(size_t i = 0; i < n; ++i) {
                if (aggr.id[i] >= 0) P->col.push_back(aggr.id[i]);
                P->ptr.push_back( static_cast<ptrdiff_t>(P->col.size()) );
            }
            P->val.resize(n, 1);
#ifdef AMGCL_HAVE_EIGEN
        }
#endif
        TOC("interpolation");

        boost::shared_ptr<Matrix> R = boost::make_shared<Matrix>();
        *R = transpose(*P);

        return boost::make_tuple(P, R);
    }

    /// Creates system matrix for the coarser level.
    /**
     * \param A The system matrix at the finer level.
     * \param P Prolongation operator returned by transfer_operators().
     * \param R Restriction operator returned by transfer_operators().
     * \returns System matrix for the coarser level.
     */
    template <class Matrix>
    static boost::shared_ptr<Matrix>
    coarse_operator(
            const Matrix &A,
            const Matrix &P,
            const Matrix &R,
            const params &prm
            )
    {
        return detail::scaled_galerkin(A, P, R, 1 / prm.over_interp);
    }
};

} // namespace coarsening
} // namespace amgcl

#endif
