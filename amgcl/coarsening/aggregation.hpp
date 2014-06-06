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
 * \brief  Coarsening by aggregation.
 */

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/scaled_galerkin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace coarsening {

template <class Aggregates>
struct aggregation {
    struct params {
        typename Aggregates::params aggr;
        float over_interp;

        params() : over_interp(1.5f) {
            aggr.eps_strong = 0.1f;
        }
    };

    template <typename Val, typename Col, typename Ptr>
    static boost::tuple<
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >,
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >
        >
    transfer_operators(
            const backend::crs<Val, Col, Ptr> &A,
            const params &prm)
    {
        typedef backend::crs<Val, Col, Ptr> matrix;

        const size_t n = rows(A);

        TIC("aggregates");
        Aggregates aggr(A, prm.aggr);
        TOC("aggregates");

        TIC("interpolation");
        boost::shared_ptr<matrix> P = boost::make_shared<matrix>();
        P->nrows = n;
        P->ncols = aggr.count;
        P->ptr.reserve(n + 1);
        P->col.reserve(n);

        P->ptr.push_back(0);
        for(size_t i = 0; i < n; ++i) {
            if (aggr.id[i] >= 0) P->col.push_back(aggr.id[i]);
            P->ptr.push_back(P->col.size());
        }
        P->val.resize(n, static_cast<Val>(1));
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
            const params &prm
            )
    {
        return detail::scaled_galerkin(A, P, R, 1 / prm.over_interp);
    }
};

} // namespace coarsening
} // namespace amgcl

#endif
