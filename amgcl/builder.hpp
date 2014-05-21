#ifndef AMGCL_BUILDER_HPP
#define AMGCL_BUILDER_HPP

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
 * \file   amgcl/builder.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Algebraic multigrid hierarchy builder.
 */

#include <iostream>
#include <iomanip>
#include <list>

#include <boost/static_assert.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/io/ios_state.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/tictoc.hpp>

namespace amgcl {

template <typename Real, class Coarsening>
struct builder {
    typedef typename backend::builtin<Real>::matrix     matrix;
    typedef typename backend::builtin<Real>::vector     vector;
    typedef typename backend::builtin<Real>::value_type value_type;
    typedef typename backend::builtin<Real>::index_type index_type;

    struct params {
        int coarse_enough;

        typename Coarsening::params coarsening;

        params() : coarse_enough(300) { }
    } prm;

    template <class Matrix>
    builder(const Matrix &M, const params &prm = params()) : prm(prm)
    {
        precondition(
                backend::rows(M) == backend::cols(M),
                "Matrix should be square!"
                );

        boost::shared_ptr<matrix> A = boost::make_shared<matrix>( M );
        boost::shared_ptr<matrix> P;
        boost::shared_ptr<matrix> R;

        while( backend::rows(*A) > prm.coarse_enough) {
            TIC("transfer operators");
            boost::tie(P, R) = Coarsening::transfer_operators(*A, prm.coarsening);
            TOC("transfer operators");

            levels.push_back( new level_fine(A, P, R) );

            TIC("coarse operator");
            A = Coarsening::coarse_operator(*A, *P, *R, prm.coarsening);
            TOC("coarse operator");

            sort_rows(*A);
        }

        boost::shared_ptr<matrix> Ainv = boost::make_shared<matrix>();
        *Ainv = inverse(*A);
        levels.push_back( new level_coarse(Ainv) );
    }

    friend std::ostream& operator<<(std::ostream &os, const builder &amg)
    {
        boost::io::ios_all_saver stream_state(os);

        size_t sum_dof = 0;
        size_t sum_nnz = 0;

        BOOST_FOREACH(const level_base &lvl, amg.levels) {
            sum_dof += lvl.rows();
            sum_nnz += lvl.nonzeros();
        }

        os << "Number of levels:    "   << amg.levels.size()
           << "\nOperator complexity: " << std::fixed << std::setprecision(2)
                                        << 1.0 * sum_nnz / amg.levels.front().nonzeros()
           << "\nGrid complexity:     " << std::fixed << std::setprecision(2)
                                        << 1.0 * sum_dof / amg.levels.front().rows()
           << "\n\nlevel     unknowns       nonzeros\n"
           << "---------------------------------\n";

        size_t depth = 0;
        BOOST_FOREACH(const level_base &lvl, amg.levels) {
            os << std::setw(5)  << depth++
               << std::setw(13) << lvl.rows()
               << std::setw(15) << lvl.nonzeros() << " ("
               << std::setw(5) << std::fixed << std::setprecision(2)
               << 100.0 * lvl.nonzeros() / sum_nnz
               << "%)" << std::endl;
        }

        return os;
    }

    struct level_base {
        boost::shared_ptr<matrix> A;

        level_base(
                boost::shared_ptr<matrix> A
                )
            : A(A)
        { }

        size_t rows() const {
            return backend::rows(*A);
        }

        size_t nonzeros() const {
            return backend::nonzeros(*A);
        }

        virtual bool is_coarse() const { return false; }

        virtual ~level_base() {};
    };

    struct level_fine : level_base {
        boost::shared_ptr<matrix> P;
        boost::shared_ptr<matrix> R;

        level_fine(
                boost::shared_ptr<matrix> A,
                boost::shared_ptr<matrix> P,
                boost::shared_ptr<matrix> R
                )
            : level_base(A), P(P), R(R)
        { }
    };

    struct level_coarse : level_base {
        level_coarse(
                boost::shared_ptr<matrix> A
                )
            : level_base(A)
        { }

        bool is_coarse() const { return true; }
    };

    boost::ptr_list< level_base > levels;

};

} // namespace amgcl

#endif
