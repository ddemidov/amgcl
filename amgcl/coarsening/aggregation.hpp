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

#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/aggregates.hpp>
#include <amgcl/tictoc.hpp>

namespace amgcl {
namespace coarsening {

struct aggregation {
    struct params {
        /// Over-interpolation factor \f$\alpha\f$.
        /**
         * See \ref Stuben_1999 "Stuben (1999)", Section 9.1 "Re-scaling of the
         * Galerkin operator". [In case of aggregation multigrid] Coarse-grid
         * correction of smooth error, and by this the overall convergence, can
         * often be substantially improved by using "over-interpolation", that
         * is, by multiplying the actual correction (corresponding to piecewise
         * constant interpolation) by some factor \f$\alpha>1\f$. Equivalently,
         * this means that the coarse-level Galerkin operator is re-scaled by
         * \f$1/\alpha\f$:
         * \f[I_h^HA_hI_H^h \to \frac{1}{\alpha}I_h^HA_hI_H^h.\f]
         */
        float over_interp;

        /// Parameter \f$\varepsilon_{str}\f$ defining strong couplings.
        /**
         * Variable \f$i\f$ is defined to be strongly coupled to another variable,
         * \f$j\f$, if \f[|a_{ij}| \geq \varepsilon_{str}\sqrt{a_{ii} a_{jj}}\quad
         * \text{with fixed} \quad 0 < \varepsilon_{str} < 1.\f]
         */
        float eps_strong;

        unsigned dof_per_node;

        params() : over_interp(1.5f), eps_strong(0.1f), dof_per_node(1) {}
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
        // TODO: deal with case of (dof > 1).
        std::vector<Col> G;
        TIC("aggregates");
        const size_t nc = detail::aggregates(A, prm.eps_strong, G);
        TOC("aggregates");

        boost::shared_ptr<matrix> P = boost::make_shared<matrix>();
        P->nrows = n;
        P->ncols = nc;
        P->ptr.reserve(n + 1);
        P->col.reserve(n);

        TIC("interpolation");
        P->ptr.push_back(0);
        for(size_t i = 0; i < n; ++i) {
            if (G[i] >= 0) P->col.push_back(G[i]);
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
        typedef backend::crs<Val, Col, Ptr> matrix;
        boost::shared_ptr<matrix> Ac = boost::make_shared<matrix>();

        *Ac = product(product(R, A), P);

        if (prm.over_interp > 1.0f)
            BOOST_FOREACH(Val &v, Ac->val) v /= prm.over_interp;

        return Ac;
    }
};

} // namespace coarsening
} // namespace amgcl

#endif
