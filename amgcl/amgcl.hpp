#ifndef AMGCL_AMGCL_HPP
#define AMGCL_AMGCL_HPP

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
 * \file   amgcl/amgcl.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Generic algebraic multigrid framework.
 */

#include <iostream>
#include <iomanip>

#include <boost/io/ios_state.hpp>
#include <boost/static_assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {

template <
    class Backend,
    class Coarsening,
    template <class> class Relax
    >
class amg {
    public:
        typedef Backend backend_type;

        typedef typename Backend::value_type value_type;
        typedef typename Backend::matrix     matrix;
        typedef typename Backend::vector     vector;
        typedef Relax<Backend>               relax_type;

        struct params {
            typedef typename Backend::params    backend_params;
            typedef typename Coarsening::params coarsening_params;
            typedef typename relax_type::params relax_params;

            backend_params    backend;
            coarsening_params coarsening;
            relax_params      relax;

            unsigned coarse_enough;

            unsigned npre;
            unsigned npost;
            unsigned ncycle;

            params() :
                coarse_enough( 300 ),
                npre         (   1 ),
                npost        (   1 ),
                ncycle       (   1 )
            {}
        } prm;

        template <class Matrix>
        amg(const Matrix &M, const params &p = params()) : prm(p)
        {
            precondition(
                    backend::rows(M) == backend::cols(M),
                    "Matrix should be square!"
                    );

            boost::shared_ptr<build_matrix> P, R;
            boost::shared_ptr<build_matrix> A = boost::make_shared<build_matrix>( M );
            sort_rows(*A);

            while( backend::rows(*A) > prm.coarse_enough) {
                TIC("transfer operators");
                boost::tie(P, R) = Coarsening::transfer_operators(
                        *A, prm.coarsening);
                TOC("transfer operators");

                TIC("move to backend")
                levels.push_back( new level(A, P, R, prm) );
                TOC("move to backend")

                TIC("coarse operator");
                A = Coarsening::coarse_operator(*A, *P, *R, prm.coarsening);
                sort_rows(*A);
                TOC("coarse operator");
            }

            TIC("coarsest level");
            boost::shared_ptr<build_matrix> Ainv = boost::make_shared<build_matrix>();
            *Ainv = inverse(*A);
            TOC("coarsest level");

            TIC("move to backend")
            levels.push_back( new level(Ainv, prm) );
            TOC("move to backend")
        }

        void cycle(const vector &rhs, vector &x) const {
            cycle(levels.begin(), rhs, x);
        }

        void operator()(const vector &rhs, vector &x) const {
            backend::clear(x);
            cycle(levels.begin(), rhs, x);
        }

        const matrix& top_matrix() const {
            return *levels.front().A;
        }
    private:
        typedef typename backend::builtin<value_type>::matrix build_matrix;

        struct level {
            boost::shared_ptr<matrix> A;
            boost::shared_ptr<matrix> P;
            boost::shared_ptr<matrix> R;

            boost::shared_ptr<vector> f;
            boost::shared_ptr<vector> u;
            boost::shared_ptr<vector> t;

            boost::shared_ptr<relax_type> relax;

            level(
                    boost::shared_ptr<build_matrix> a,
                    boost::shared_ptr<build_matrix> p,
                    boost::shared_ptr<build_matrix> r,
                    const params &prm
                 ) :
                A( Backend::copy_matrix(a, prm.backend) ),
                P( Backend::copy_matrix(p, prm.backend) ),
                R( Backend::copy_matrix(r, prm.backend) ),
                f( Backend::create_vector(backend::rows(*a), prm.backend) ),
                u( Backend::create_vector(backend::rows(*a), prm.backend) ),
                t( Backend::create_vector(backend::rows(*a), prm.backend) ),
                relax( new relax_type(*a, prm.relax, prm.backend) )
            { }

            level(
                    boost::shared_ptr<build_matrix> a,
                    const params &prm
                 ) :
                A( Backend::copy_matrix(a, prm.backend) ),
                f( Backend::create_vector(backend::rows(*a), prm.backend) ),
                u( Backend::create_vector(backend::rows(*a), prm.backend) )
            { }

            size_t rows() const {
                return backend::rows(*A);
            }

            size_t nonzeros() const {
                return backend::nonzeros(*A);
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
                for (size_t j = 0; j < prm.ncycle; ++j) {
                    TIC("relax");
                    for(size_t i = 0; i < prm.npre; ++i)
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
                    for(size_t i = 0; i < prm.npre; ++i)
                        lvl->relax->apply_post(*lvl->A, rhs, x, *lvl->t, prm.relax);
                    TOC("relax");
                }
            }
        }

    friend std::ostream& operator<<(std::ostream &os, const amg &a)
    {
        boost::io::ios_all_saver stream_state(os);

        size_t sum_dof = 0;
        size_t sum_nnz = 0;

        BOOST_FOREACH(const level &lvl, a.levels) {
            sum_dof += lvl.rows();
            sum_nnz += lvl.nonzeros();
        }

        os << "Number of levels:    "   << a.levels.size()
           << "\nOperator complexity: " << std::fixed << std::setprecision(2)
                                        << 1.0 * sum_nnz / a.levels.front().nonzeros()
           << "\nGrid complexity:     " << std::fixed << std::setprecision(2)
                                        << 1.0 * sum_dof / a.levels.front().rows()
           << "\n\nlevel     unknowns       nonzeros\n"
           << "---------------------------------\n";

        size_t depth = 0;
        BOOST_FOREACH(const level &lvl, a.levels) {
            os << std::setw(5)  << depth++
               << std::setw(13) << lvl.rows()
               << std::setw(15) << lvl.nonzeros() << " ("
               << std::setw(5) << std::fixed << std::setprecision(2)
               << 100.0 * lvl.nonzeros() / sum_nnz
               << "%)" << std::endl;
        }

        return os;
    }

};

} // namespace amgcl

#endif
