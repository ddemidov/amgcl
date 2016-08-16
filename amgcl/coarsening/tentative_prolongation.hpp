#ifndef AMGCL_COARSENING_TENTATIVE_PROLONGATION_HPP
#define AMGCL_COARSENING_TENTATIVE_PROLONGATION_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/coarsening/tentative_prolongation.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Tentative prolongation operator for aggregated AMG.
 */

#include <vector>
#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/multi_array.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/detail/qr.hpp>

namespace amgcl {
namespace coarsening {
namespace detail {
    struct skip_negative {
        const std::vector<ptrdiff_t> &key;
        int block_size;

        skip_negative(const std::vector<ptrdiff_t> &key, int block_size)
            : key(key), block_size(block_size) { }

        bool operator()(ptrdiff_t i, ptrdiff_t j) const {
            // Cast to unsigned type to keep negative values at the end
            return
                static_cast<size_t>(key[i]) / block_size <
                static_cast<size_t>(key[j]) / block_size;
        }
    };
} // namespace detail


/// Tentative prolongation operator
/**
 * If near nullspace vectors are not provided, returns piecewise-constant
 * prolongation operator. If user provides near nullspace vectors, those are
 * used to improve the prolongation operator.
 * \see \cite Vanek2001
 */
template <class Matrix>
boost::shared_ptr<Matrix> tentative_prolongation(
        size_t n,
        size_t naggr,
        const std::vector<ptrdiff_t> aggr,
        boost::multi_array<typename math::rhs_of<typename backend::value_type<Matrix>::type>::type, 2> &B,
        int block_size
        )
{
    typedef typename backend::value_type<Matrix>::type value_type;

    boost::shared_ptr<Matrix> P = boost::make_shared<Matrix>();

    TIC("tentative");
    if (int nvec = boost::size(B)) {
        // Sort fine points by aggregate number.
        // Put points not belonging to any aggregate to the end of the list.
        std::vector<ptrdiff_t> order(
                boost::counting_iterator<ptrdiff_t>(0),
                boost::counting_iterator<ptrdiff_t>(n)
                );
        boost::stable_sort(order, detail::skip_negative(aggr, block_size));

        // Precompute the shape of the prolongation operator.
        // Each row contains exactly nvec non-zero entries.
        // Rows that do not belong to any aggregate are empty.
        P->nrows = n;
        P->ncols = nvec * naggr / block_size;
        P->ptr.reserve(n + 1);

        P->ptr.push_back(0);
        for(size_t i = 0; i < n; ++i) {
            if (aggr[i] < 0)
                P->ptr.push_back(P->ptr.back());
            else
                P->ptr.push_back(P->ptr.back() + nvec);
        }

        P->col.resize(P->ptr.back());
        P->val.resize(P->ptr.back());

        // Compute the tentative prolongation operator and null-space vectors
        // for the coarser level.
        boost::multi_array<typename math::rhs_of<value_type>::type, 2> Bnew(
                boost::extents[naggr * nvec / block_size][nvec],
                boost::fortran_storage_order()
                );

        size_t offset = 0, Bcol = 0;

        amgcl::detail::QR<value_type, amgcl::detail::col_major> qr;
        std::vector<value_type> Bpart;
        for(ptrdiff_t i = 0, nb = naggr / block_size; i < nb; ++i) {
            size_t d = 0;
            for(size_t j = offset; j < n && aggr[order[j]] / block_size == i; ++j, ++d);
            Bpart.resize(d * nvec);

            for(size_t j = offset, jj = 0; jj < d; ++j, ++jj) {
                for(int k = 0; k < nvec; ++k)
                    Bpart[jj + d * k] = B[k][order[j]];
            }

            qr.compute(d, nvec, &Bpart[0]);
            qr.compute_q();

            for(int ii = 0; ii < nvec; ++ii, ++Bcol)
                for(int jj = 0; jj < nvec; ++jj)
                    Bnew[jj][Bcol] = qr.R(ii,jj);

            for(size_t ii = 0; ii < d; ++ii, ++offset) {
                ptrdiff_t  *c = &P->col[P->ptr[order[offset]]];
                value_type *v = &P->val[P->ptr[order[offset]]];

                for(int jj = 0; jj < nvec; ++jj) {
                    c[jj] = i * nvec + jj;
                    v[jj] = qr.Q(ii,jj);
                }
            }
        }

        // TODO: make this more effective
        //B.resize(Bnew.shape());
        //B = Bnew;
    } else {
        P->nrows = n;
        P->ncols = naggr;
        P->ptr.reserve(n + 1);
        P->col.reserve(n);

        P->ptr.push_back(0);
        for(size_t i = 0; i < n; ++i) {
            if (aggr[i] >= 0) P->col.push_back(aggr[i]);
            P->ptr.push_back( static_cast<ptrdiff_t>(P->col.size()) );
        }
        P->val.resize(n, math::identity<value_type>());
    }
    TOC("tentative");

    return P;
}

} // namespace coarsening
} // namespace amgcl

#endif
