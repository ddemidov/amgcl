#ifndef AMGCL_SOLVER_HPP
#define AMGCL_SOLVER_HPP

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
 * \file   amgcl/solver.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Solver/preconditioner based on algebraic multigrid method.
 */

#include <boost/ptr_container/ptr_list.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/relaxation/interface.hpp>
#include <amgcl/tictoc.hpp>

namespace amgcl {

template <class Backend, relaxation::scheme Relax>
class amg {
    public:
        typedef typename Backend::matrix matrix;
        typedef typename Backend::vector vector;
        typedef typename Backend::value_type value_type;
        typedef relaxation::impl<Relax, Backend> relax_type;

        struct params {
            typename Backend::params    backend;
            typename relax_type::params relax;

            int ncycle;
            int npre;
            int npost;

            params() : ncycle(1), npre(1), npost(1) {}
        } prm;

        template <class Builder>
        amg(const Builder &builder, const params &prm = params())
            : prm(prm)
        {
            typedef typename Builder::level_base   bl_base;
            typedef typename Builder::level_coarse bl_coarse;
            typedef typename Builder::level_fine   bl_fine;

            BOOST_FOREACH(const bl_base &lvl, builder.levels)
            {
                if (lvl.is_coarse())
                    levels.push_back( new level(
                                dynamic_cast<const bl_coarse&>(lvl),
                                prm, boost::true_type() ) );
                else
                    levels.push_back( new level(
                                dynamic_cast<const bl_fine&>(lvl),
                                prm, boost::false_type() ) );
            }
        }

        void apply(const vector &rhs, vector &x) const {
            cycle(levels.begin(), rhs, x);
        }

        void operator()(const vector &rhs, vector &x) const {
            apply(rhs, x);
        }

        const matrix& top_matrix() const {
            return *levels.front().A;
        }
    private:
        struct level {
            boost::shared_ptr<matrix> A;
            boost::shared_ptr<matrix> P;
            boost::shared_ptr<matrix> R;

            boost::shared_ptr<vector> f;
            boost::shared_ptr<vector> u;
            boost::shared_ptr<vector> t;

            boost::shared_ptr<relax_type> relax;

            template <class Builder>
            level(const Builder &builder, const params &prm,
                    boost::false_type)
            {
                A = Backend::copy_matrix(builder.A, prm.backend);
                P = Backend::copy_matrix(builder.P, prm.backend);
                R = Backend::copy_matrix(builder.R, prm.backend);

                f = Backend::create_vector(builder.rows(), prm.backend);
                u = Backend::create_vector(builder.rows(), prm.backend);
                t = Backend::create_vector(builder.rows(), prm.backend);

                relax = boost::make_shared<relax_type>(*builder.A, prm.relax, prm.backend);
            }

            template <class Builder>
            level(const Builder &builder, const params &prm,
                    boost::true_type)
            {
                A = Backend::copy_matrix(builder.A, prm.backend);

                f = Backend::create_vector(builder.rows(), prm.backend);
                u = Backend::create_vector(builder.rows(), prm.backend);
            }
        };

        typedef typename boost::ptr_list<level>::const_iterator level_iterator;

        boost::ptr_list<level> levels;

        void cycle(level_iterator lvl, const vector &rhs, vector &x) const
        {
            level_iterator nxt = lvl; ++nxt;

            if (nxt == levels.end()) {
                TIC("coarse");
                backend::spmv(1, *lvl->A, rhs, 0, x);
                TOC("coarse");
            } else {
                for (int j = 0; j < prm.ncycle; ++j) {
                    TIC("relax");
                    for(int i = 0; i < prm.npre; ++i)
                        lvl->relax->apply_pre(*lvl->A, rhs, x, *lvl->t, prm.relax);
                    TOC("relax");

                    TIC("residual");
                    backend::residual(rhs, *lvl->A, x, *lvl->t);
                    TOC("residual");

                    TIC("restrict");
                    backend::spmv(1, *lvl->R, *lvl->t, 0, *nxt->f);
                    TOC("restrict");

                    backend::clear(*nxt->u);
                    cycle(nxt, *nxt->f, *nxt->u);

                    TIC("prolongate");
                    backend::spmv(1, *lvl->P, *nxt->u, 1, x);
                    TOC("prolongate");

                    TIC("relax");
                    for(int i = 0; i < prm.npre; ++i)
                        lvl->relax->apply_post(*lvl->A, rhs, x, *lvl->t, prm.relax);
                    TOC("relax");
                }
            }
        }
};

} // namespace amgcl

#endif
