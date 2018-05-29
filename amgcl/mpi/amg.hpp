#ifndef AMGCL_MPI_AMG_HPP
#define AMGCL_MPI_AMG_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/mpi/amg.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed memory AMG preconditioner.
 */

#include <iostream>
#include <iomanip>
#include <list>

#include <boost/io/ios_state.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/mpi/direct_solver/skyline_lu.hpp>
#include <amgcl/mpi/repartition/dummy.hpp>

namespace amgcl {
namespace mpi {

template <
    class Backend,
    class Coarsening,
    class Relaxation,
    class DirectSolver = direct::skyline_lu<typename Backend::value_type>,
    class Repartition = repartition::dummy<Backend>
    >
class amg {
    public:
        typedef Backend                                    backend_type;
        typedef typename Backend::params                   backend_params;
        typedef typename Backend::value_type               value_type;
        typedef typename math::scalar_of<value_type>::type scalar_type;
        typedef distributed_matrix<Backend>                matrix;
        typedef typename Backend::vector                   vector;

        struct params {
            typedef typename Coarsening::params   coarsening_params;
            typedef typename Relaxation::params   relax_params;
            typedef typename DirectSolver::params direct_params;
            typedef typename Repartition::params  repart_params;

            coarsening_params coarsening;   ///< Coarsening parameters.
            relax_params      relax;        ///< Relaxation parameters.
            direct_params     direct;       ///< Direct solver parameters.
            repart_params     repart;       ///< Repartition parameters.

            /// Specifies when level is coarse enough to be solved directly.
            /**
             * If number of variables at a next level in the hierarchy becomes
             * lower than this threshold, then the hierarchy construction is
             * stopped and the linear system is solved directly at this level.
             */
            unsigned coarse_enough;

            /// Use direct solver at the coarsest level.
            /**
             * When set, the coarsest level is solved with a direct solver.
             * Otherwise a smoother is used as a solver.
             */
            bool direct_coarse;

            /// Maximum number of levels.
            /** If this number is reached while the size of the last level is
             * greater that `coarse_enough`, then the coarsest level will not
             * be solved exactly, but will use a smoother.
             */
            unsigned max_levels;

            /// Number of pre-relaxations.
            unsigned npre;

            /// Number of post-relaxations.
            unsigned npost;

            /// Number of cycles (1 for V-cycle, 2 for W-cycle, etc.).
            unsigned ncycle;

            /// Number of cycles to make as part of preconditioning.
            unsigned pre_cycles;

            params() :
                coarse_enough(DirectSolver::coarse_enough()), direct_coarse(true),
                max_levels( std::numeric_limits<unsigned>::max() ),
                npre(1), npost(1), ncycle(1), pre_cycles(1)
            {}

            params(const boost::property_tree::ptree &p)
                : AMGCL_PARAMS_IMPORT_CHILD(p, coarsening),
                  AMGCL_PARAMS_IMPORT_CHILD(p, relax),
                  AMGCL_PARAMS_IMPORT_CHILD(p, direct),
                  AMGCL_PARAMS_IMPORT_CHILD(p, repart),
                  AMGCL_PARAMS_IMPORT_VALUE(p, coarse_enough),
                  AMGCL_PARAMS_IMPORT_VALUE(p, direct_coarse),
                  AMGCL_PARAMS_IMPORT_VALUE(p, max_levels),
                  AMGCL_PARAMS_IMPORT_VALUE(p, npre),
                  AMGCL_PARAMS_IMPORT_VALUE(p, npost),
                  AMGCL_PARAMS_IMPORT_VALUE(p, ncycle),
                  AMGCL_PARAMS_IMPORT_VALUE(p, pre_cycles)
            {
                AMGCL_PARAMS_CHECK(p, (coarsening)(relax)(direct)(repart)(coarse_enough)
                        (direct_coarse)(max_levels)(npre)(npost)(ncycle)(pre_cycles));

                amgcl::precondition(max_levels > 0, "max_levels should be positive");
            }

            void get(
                    boost::property_tree::ptree &p,
                    const std::string &path = ""
                    ) const
            {
                AMGCL_PARAMS_EXPORT_CHILD(p, path, coarsening);
                AMGCL_PARAMS_EXPORT_CHILD(p, path, relax);
                AMGCL_PARAMS_EXPORT_CHILD(p, path, direct);
                AMGCL_PARAMS_EXPORT_CHILD(p, path, repart);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, coarse_enough);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, direct_coarse);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, max_levels);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, npre);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, npost);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, ncycle);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, pre_cycles);
            }
        } prm;

        template <class Matrix>
        amg(
                communicator comm,
                const Matrix &A,
                const params &prm = params(),
                const backend_params &bprm = backend_params()
           ) : prm(prm), repart(prm.repart)
        {
            init(boost::make_shared<matrix>(comm, A, backend::rows(A)), bprm);
        }

        amg(
                communicator,
                boost::shared_ptr<matrix> A,
                const params &prm = params(),
                const backend_params &bprm = backend_params()
           ) : prm(prm), repart(prm.repart)
        {
            init(A, bprm);
        }

        template <class Vec1, class Vec2>
        void cycle(
                const Vec1 &rhs,
#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
                Vec2       &x
#else
                Vec2       &&x
#endif
                ) const
        {
            cycle(levels.begin(), rhs, x);
        }

        template <class Vec1, class Vec2>
        void apply(
                const Vec1 &rhs,
#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
                Vec2       &x
#else
                Vec2       &&x
#endif
                ) const
        {
            if (prm.pre_cycles) {
                backend::clear(x);
                for(unsigned i = 0; i < prm.pre_cycles; ++i)
                    cycle(levels.begin(), rhs, x);
            } else {
                backend::copy(rhs, x);
            }
        }

        /// Returns the system matrix from the finest level.
        boost::shared_ptr<matrix> system_matrix_ptr() const {
            return A;
        }

        const matrix& system_matrix() const {
            return *system_matrix_ptr();
        }
    private:
        struct level {
            ptrdiff_t nrows, nnz;
            int active_procs;

            boost::shared_ptr<matrix>       A, P, R;
            boost::shared_ptr<vector>       f, u, t;
            boost::shared_ptr<Relaxation>   relax;
            boost::shared_ptr<DirectSolver> solve;

            level() {}

            level(
                    boost::shared_ptr<matrix> a,
                    params &prm,
                    const backend_params &bprm,
                    bool direct = false
                 )
                : nrows(a->glob_rows()), nnz(a->glob_nonzeros()),
                  f(Backend::create_vector(a->loc_rows(), bprm)),
                  u(Backend::create_vector(a->loc_rows(), bprm))
            {
                int active = (a->loc_rows() > 0);
                MPI_Allreduce(&active, &active_procs, 1, MPI_INT, MPI_SUM, a->comm());

                sort_rows(*a->local());
                sort_rows(*a->remote());

                if (direct) {
                    AMGCL_TIC("direct solver");
                    solve = boost::make_shared<DirectSolver>(a->comm(), *a, prm.direct);
                    AMGCL_TOC("direct solver");
                } else {
                    A = a;
                    t = Backend::create_vector(a->loc_rows(), bprm);

                    AMGCL_TIC("relaxation");
                    relax = boost::make_shared<Relaxation>(*a, prm.relax, bprm);
                    AMGCL_TOC("relaxation");
                }
            }

            boost::shared_ptr<matrix> step_down(Coarsening &C, const Repartition &repart)
            {
                AMGCL_TIC("transfer operators");
                boost::tie(P, R) = C.transfer_operators(*A);

                AMGCL_TIC("sort");
                sort_rows(*P->local());
                sort_rows(*P->remote());
                sort_rows(*R->local());
                sort_rows(*R->remote());
                AMGCL_TOC("sort");

                AMGCL_TOC("transfer operators");

                if (P->glob_cols() == 0) {
                    // Zero-sized coarse level in amgcl (diagonal matrix?)
                    return boost::shared_ptr<matrix>();
                }

                AMGCL_TIC("coarse operator");
                boost::shared_ptr<matrix> Ac = C.coarse_operator(*A, *P, *R);
                AMGCL_TOC("coarse operator");

                if (repart.is_needed(*Ac)) {
                    boost::shared_ptr<matrix> I = repart(*Ac);
                    boost::shared_ptr<matrix> J = transpose(*I);

                    P  = product(*P, *I);
                    R  = product(*J, *R);
                    Ac = product(*J, *product(*Ac, *I));
                }

                return Ac;
            }

            void move_to_backend(const backend_params &bprm) {
                AMGCL_TIC("move to backend");
                if (A) A->move_to_backend(bprm);
                if (P) P->move_to_backend(bprm);
                if (R) R->move_to_backend(bprm);
                AMGCL_TOC("move to backend");
            }

            ptrdiff_t rows() const {
                return nrows;
            }

            ptrdiff_t nonzeros() const {
                return nnz;
            }
        };

        typedef typename std::list<level>::const_iterator level_iterator;

        boost::shared_ptr<matrix> A;
        Repartition repart;
        std::list<level> levels;

        void init(boost::shared_ptr<matrix> A, const backend_params &bprm)
        {
            mpi::precondition(A->comm(), A->glob_rows() == A->glob_cols(),
                    "Matrix should be square!");

            this->A = A;
            Coarsening C(prm.coarsening);
            bool direct_coarse = prm.direct_coarse;

            while(A->glob_rows() > prm.coarse_enough && levels.size() < prm.max_levels) {
                levels.push_back( level(A, prm, bprm) );
                if (levels.size() >= prm.max_levels) break;

                A = levels.back().step_down(C, repart);
                levels.back().move_to_backend(bprm);

                if (!A) {
                    // Zero-sized coarse level. Probably the system matrix on
                    // this level is diagonal, should be easily solvable with a
                    // couple of smoother iterations.
                    direct_coarse = false;
                    break;
                }
            }

            if (!A || A->loc_rows() > prm.coarse_enough) {
                // The coarse matrix is still too big to be solved directly.
                direct_coarse = false;
            }

            if (A) {
                levels.push_back(level(A, prm, bprm, direct_coarse));
                levels.back().move_to_backend(bprm);
            }

            AMGCL_TIC("move to backend");
            this->A->move_to_backend(bprm);
            AMGCL_TOC("move to backend");
        }

        template <class Vec1, class Vec2>
        void cycle(level_iterator lvl, const Vec1 &rhs, Vec2 &x) const {
            level_iterator nxt = lvl, end = levels.end();
            ++nxt;

            if (nxt == end) {
                if (lvl->solve) {
                    AMGCL_TIC("direct solver");
                    (*lvl->solve)(rhs, x);
                    AMGCL_TOC("direct solver");
                } else {
                    AMGCL_TIC("relax");
                    lvl->relax->apply_pre(*lvl->A, rhs, x, *lvl->t);
                    lvl->relax->apply_post(*lvl->A, rhs, x, *lvl->t);
                    AMGCL_TOC("relax");
                }
            } else {
                for (size_t j = 0; j < prm.ncycle; ++j) {
                    AMGCL_TIC("relax");
                    for(size_t i = 0; i < prm.npre; ++i)
                        lvl->relax->apply_pre(*lvl->A, rhs, x, *lvl->t);
                    AMGCL_TOC("relax");

                    backend::residual(rhs, *lvl->A, x, *lvl->t);

                    backend::spmv(math::identity<scalar_type>(), *lvl->R, *lvl->t, math::zero<scalar_type>(), *nxt->f);

                    backend::clear(*nxt->u);
                    cycle(nxt, *nxt->f, *nxt->u);

                    backend::spmv(math::identity<scalar_type>(), *lvl->P, *nxt->u, math::identity<scalar_type>(), x);

                    AMGCL_TIC("relax");
                    for(size_t i = 0; i < prm.npost; ++i)
                        lvl->relax->apply_post(*lvl->A, rhs, x, *lvl->t);
                    AMGCL_TOC("relax");
                }
            }
        }

    template <class B, class C, class R, class D, class I>
    friend std::ostream& operator<<(std::ostream &os, const amg<B, C, R, D, I> &a);
};

template <class B, class C, class R, class D, class I>
std::ostream& operator<<(std::ostream &os, const amg<B, C, R, D, I> &a)
{
    typedef typename amg<B, C, R, D, I>::level level;
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
           << "%) [" << lvl.active_procs << "]" << std::endl;
    }

    return os;
}

} // namespace mpi
} // namespace amgcl

#endif
