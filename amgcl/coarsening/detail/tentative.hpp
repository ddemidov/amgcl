#ifndef AMGCL_COARSENING_DETAIL_TENTATIVE_HPP
#define AMGCL_COARSENING_DETAIL_TENTATIVE_HPP

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
 * \file   amgcl/coarsening/detail/tentative.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Tentative prolongation operator for aggregated AMG.
 */

#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/algorithm.hpp>

#include <Eigen/Dense>
#include <Eigen/QR>

namespace amgcl {
namespace coarsening {
namespace detail {

//---------------------------------------------------------------------------
template <class Matrix>
boost::shared_ptr<Matrix> tentative_prolongation(
        size_t n,
        size_t naggr,
        const std::vector<ptrdiff_t> aggr
        )
{
    typedef typename backend::value_type<Matrix>::type value_type;

    boost::shared_ptr<Matrix> P = boost::make_shared<Matrix>();

    P->nrows = n;
    P->ncols = naggr;
    P->ptr.reserve(n + 1);
    P->col.reserve(n);

    P->ptr.push_back(0);
    for(size_t i = 0; i < n; ++i) {
        if (aggr[i] >= 0) P->col.push_back(aggr[i]);
        P->ptr.push_back( static_cast<ptrdiff_t>(P->col.size()) );
    }
    P->val.resize(n, static_cast<value_type>(1));

    return P;
}

//---------------------------------------------------------------------------
struct skip_negative {
    const std::vector<ptrdiff_t> &key;

    skip_negative(const std::vector<ptrdiff_t> &key) : key(key) { }

    bool operator()(ptrdiff_t i, ptrdiff_t j) const {
        // Cast to unsigned type to keep negative values at the end
        return
            static_cast<size_t>(key[i]) <
            static_cast<size_t>(key[j]);
    }
};

//---------------------------------------------------------------------------
template <class Matrix>
boost::tuple<
    boost::shared_ptr<Matrix>,
    std::vector<double>
    >
tentative_prolongation(
        size_t n,
        size_t naggr,
        const std::vector<ptrdiff_t> aggr,
        int Bcols,
        const std::vector<double> &B
        )
{
    typedef typename backend::value_type<Matrix>::type value_type;

    boost::shared_ptr<Matrix> P = boost::make_shared<Matrix>();
    std::vector<double> Bnew;

    std::vector<ptrdiff_t> order(
            boost::counting_iterator<ptrdiff_t>(0),
            boost::counting_iterator<ptrdiff_t>(n)
            );

    boost::sort(order, skip_negative(aggr));

    P->nrows = n;
    P->ncols = naggr * Bcols;
    P->ptr.reserve(n + 1);

    P->ptr.push_back(0);
    for(size_t i = 0; i < n; ++i) {
        if (aggr[i] < 0)
            P->ptr.push_back(P->ptr.back());
        else
            P->ptr.push_back(P->ptr.back() + Bcols);
    }

    P->col.resize(P->ptr.back());
    P->val.resize(P->ptr.back());

    Bnew.reserve(naggr * Bcols * Bcols);

    typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EMatrix;
    typedef Eigen::Map<EMatrix> EMap;

    size_t offset = 0;

    std::vector<double> Bdata;
    for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(naggr); ++i) {
        int d = 0;
        Bdata.clear();
        for(ptrdiff_t j = offset; j < n && aggr[order[j]] == i; ++j, ++d)
            std::copy(
                    &B[Bcols * order[j]], &B[Bcols * (order[j] + 1)],
                    std::back_inserter(Bdata)
                    );

        EMap Bpart(Bdata.data(), d, Bcols);
        Eigen::HouseholderQR<EMatrix> qr(Bpart);

        EMatrix R = qr.matrixQR().template triangularView<Eigen::Upper>();
        EMatrix Q = qr.householderQ();

        double sign = R(0,0) > 0 ? 1 : -1;
        for(int ii = 0; ii < Bcols; ++ii)
            for(int jj = 0; jj < Bcols; ++jj)
                Bnew.push_back( R(ii,jj) * sign );

        for(int ii = 0; ii < d; ++ii, ++offset) {
            ptrdiff_t  *c = &P->col[P->ptr[order[offset]]];
            value_type *v = &P->val[P->ptr[order[offset]]];

            for(int jj = 0; jj < Bcols; ++jj) {
                c[jj] = i * Bcols + jj;
                v[jj] = Q(ii,jj) * sign;
            }
        }
    }

    return boost::make_tuple(P, Bnew);
}

} // namespace detail
} // namespace coarsening
} // namespace amgcl

#endif
