#ifndef AMGCL_AMG_HPP
#define AMGCL_AMG_HPP

/*
The MIT License

Copyright (c) 2012-2019 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/amg.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  An AMG preconditioner.
 */

#include <iostream>
#include <iomanip>
#include <list>
#include <memory>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/solver/detail/default_inner_product.hpp>
#include <amgcl/util.hpp>

/// Primary namespace.
namespace amgcl {

/// Algebraic multigrid method.
/**
 * AMG is one the most effective methods for solution of large sparse
 * unstructured systems of equations, arising, for example, from discretization
 * of PDEs on unstructured grids \cite Trottenberg2001. The method can be used
 * as a black-box solver for various computational problems, since it does not
 * require any information about the underlying geometry.
 *
 * The three template parameters allow the user to select the exact components
 * of the method:
 *  1. *Backend* to transfer the constructed hierarchy to,
 *  2. *Coarsening* strategy for hierarchy construction, and
 *  3. *Relaxation* scheme (smoother to use during the solution phase).
 *
 * Instance of the class builds the AMG hierarchy for the given system matrix
 * and is intended to be used as a preconditioner.
 */
template <
    class Backend,
    template <class> class Coarsening,
    template <class> class Relax
    >
class amg {
    public:
        typedef Backend backend_type;

        typedef typename Backend::value_type value_type;
        typedef typename Backend::matrix     matrix;
        typedef typename Backend::vector     vector;

        typedef Coarsening<Backend>          coarsening_type;
        typedef Relax<Backend>               relax_type;

        typedef typename backend::builtin<value_type>::matrix build_matrix;

        typedef typename math::scalar_of<value_type>::type scalar_type;

        /// Backend parameters.
        typedef typename Backend::params     backend_params;

        /// Parameters of the method.
        /**
         * The amgcl::amg::params struct includes parameters for each
         * component of the method as well as some universal parameters.
         */
        struct params {
            typedef typename coarsening_type::params coarsening_params;
            typedef typename relax_type::params relax_params;

            coarsening_params coarsening;   ///< Coarsening parameters.
            relax_params      relax;        ///< Relaxation parameters.

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
                coarse_enough( Backend::direct_solver::coarse_enough() ),
                direct_coarse(true),
                max_levels( std::numeric_limits<unsigned>::max() ),
                npre(1), npost(1), ncycle(1), pre_cycles(1)
            {}

#ifndef AMGCL_NO_BOOST
            params(const boost::property_tree::ptree &p)
                : AMGCL_PARAMS_IMPORT_CHILD(p, coarsening),
                  AMGCL_PARAMS_IMPORT_CHILD(p, relax),
                  AMGCL_PARAMS_IMPORT_VALUE(p, coarse_enough),
                  AMGCL_PARAMS_IMPORT_VALUE(p, direct_coarse),
                  AMGCL_PARAMS_IMPORT_VALUE(p, max_levels),
                  AMGCL_PARAMS_IMPORT_VALUE(p, npre),
                  AMGCL_PARAMS_IMPORT_VALUE(p, npost),
                  AMGCL_PARAMS_IMPORT_VALUE(p, ncycle),
                  AMGCL_PARAMS_IMPORT_VALUE(p, pre_cycles)
            {
                check_params(p, {"coarsening", "relax", "coarse_enough",
                        "direct_coarse", "max_levels", "npre", "npost",
                        "ncycle",  "pre_cycles"
                        });

                precondition(max_levels > 0, "max_levels should be positive");
            }

            void get(
                    boost::property_tree::ptree &p,
                    const std::string &path = ""
                    ) const
            {
                AMGCL_PARAMS_EXPORT_CHILD(p, path, coarsening);
                AMGCL_PARAMS_EXPORT_CHILD(p, path, relax);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, coarse_enough);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, direct_coarse);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, max_levels);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, npre);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, npost);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, ncycle);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, pre_cycles);
            }
#endif
        } prm;

        /// Builds the AMG hierarchy for the system matrix.
        /**
         * The input matrix is copied here and is safe to delete afterwards.
         *
         * \param A The system matrix. Should be convertible to
         *          amgcl::backend::crs<>.
         * \param p AMG parameters.
         *
         * \sa amgcl/adapter/crs_tuple.hpp
         */
        template <class Matrix>
        amg(
                const Matrix &A,
                const params &p = params(),
                const backend_params &bprm = backend_params()
           ) : prm(p)
        {
            precondition(backend::rows(A) == backend::cols(A), "Matrix should be square!");
            coarsening_type C(prm.coarsening);
            add_level(A, C, bprm, first_level{});
        }

        /// Performs single V-cycle for the given right-hand side and solution.
        /**
         * \param A   System matrix at the top level.
         * \param rhs Right-hand side vector.
         * \param x   Solution vector.
         */
        template <class Matrix, class Vec1, class Vec2>
        void cycle(const Matrix &A, const Vec1 &rhs, Vec2 &&x) const {
            cycle(&A, levels.begin(), rhs, x);
        }

        /// Performs single V-cycle after clearing x.
        /**
         * This is intended for use as a preconditioning procedure.
         *
         * \param A   System matrix at the top level.
         * \param rhs Right-hand side vector.
         * \param x   Solution vector.
         */
        template <class Matrix, class Vec1, class Vec2>
        void apply(const Matrix &A, const Vec1 &rhs, Vec2 &&x) const {
            if (prm.pre_cycles) {
                backend::clear(x);
                for(unsigned i = 0; i < prm.pre_cycles; ++i)
                    cycle(A, rhs, x);
            } else {
                backend::copy(rhs, x);
            }
        }

        size_t bytes() const {
            size_t b = 0;
            for(const auto &lvl : levels) b += lvl.bytes();
            return b;
        }
    private:
        struct first_level {};

        struct level {
            size_t m_rows, m_nonzeros;

            std::shared_ptr<vector> f;
            std::shared_ptr<vector> u;
            std::shared_ptr<vector> t;

            std::shared_ptr<matrix> A;
            std::shared_ptr<matrix> P;
            std::shared_ptr<matrix> R;

            std::shared_ptr< typename Backend::direct_solver > solve;

            std::shared_ptr<relax_type> relax;

            size_t bytes() const {
                size_t b = 0;

                if (f) b += backend::bytes(*f);
                if (u) b += backend::bytes(*u);
                if (t) b += backend::bytes(*t);

                if (A) b += backend::bytes(*A);
                if (P) b += backend::bytes(*P);
                if (R) b += backend::bytes(*R);

                if (solve) b += backend::bytes(*solve);
                if (relax) b += backend::bytes(*relax);

                return b;
            }

            template <class Matrix, class MatrixPtr>
            level(const Matrix &A, params &prm, const backend_params &bprm, MatrixPtr Aptr)
                : m_rows(backend::rows(A)), m_nonzeros(backend::nonzeros(A))
            {
                AMGCL_TIC("move to backend");
                f = Backend::create_vector(m_rows, bprm);
                u = Backend::create_vector(m_rows, bprm);
                AMGCL_TOC("move to backend");

                if (m_rows <= prm.coarse_enough && prm.direct_coarse) {
                    AMGCL_TIC("direct solver");
                    solve = Backend::create_solver(A, bprm);
                    AMGCL_TOC("direct solver");
                } else {
                    AMGCL_TIC("move to backend");
                    t = Backend::create_vector(m_rows, bprm);
                    AMGCL_TOC("move to backend");

                    AMGCL_TIC("relaxation");
                    relax = std::make_shared<relax_type>(A, prm.relax, bprm);
                    AMGCL_TOC("relaxation");
                }

                store_matrix(Aptr, bprm);
            }

            void store_matrix(first_level, const backend_params&) {}

            void store_matrix(std::shared_ptr<build_matrix> ptr, const backend_params &bprm) {
                AMGCL_TIC("move to backend");
                A = Backend::copy_matrix(ptr, bprm);
                AMGCL_TOC("move to backend");
            }

            template <class Matrix>
            std::shared_ptr<build_matrix> step_down(
                    const Matrix &A, coarsening_type &C, const backend_params &bprm)
            {
                AMGCL_TIC("transfer operators");
                auto P = std::make_shared<build_matrix>();
                auto R = std::make_shared<build_matrix>();

                try {
                    C.transfer_operators(A, *P, *R);
                } catch(error::empty_level) {
                    AMGCL_TOC("transfer operators");
                    return std::shared_ptr<build_matrix>();
                }

                sort_rows(*P);
                sort_rows(*R);
                AMGCL_TOC("transfer operators");

                AMGCL_TIC("move to backend");
                this->P = Backend::copy_matrix(P, bprm);
                this->R = Backend::copy_matrix(R, bprm);
                AMGCL_TOC("move to backend");

                AMGCL_TIC("coarse operator");
                auto Ac = std::make_shared<build_matrix>();
                C.coarse_operator(A, *P, *R, *Ac);
                sort_rows(*Ac);
                AMGCL_TOC("coarse operator");

                return Ac;
            }

            size_t rows() const {
                return m_rows;
            }

            size_t nonzeros() const {
                return m_nonzeros;
            }
        };

        typedef typename std::list<level>::const_iterator level_iterator;

        std::list<level> levels;

        template <class Matrix, class MatrixPtr>
        void add_level(
                const Matrix &A,
                coarsening_type &C,
                const backend_params &bprm,
                MatrixPtr Aptr)
        {
            levels.emplace_back(A, prm, bprm, Aptr);

            if (backend::rows(A) <= prm.coarse_enough || levels.size() >= prm.max_levels)
                return;

            auto Ac = levels.back().step_down(A, C, bprm);
            if (Ac) add_level(*Ac, C, bprm, Ac);
        }

        template <class Matrix, class Vec1, class Vec2>
        void cycle(const Matrix A, level_iterator lvl, const Vec1 &rhs, Vec2 &x) const
        {
            level_iterator nxt = lvl, end = levels.end();
            ++nxt;

            if (nxt == end) {
                if (lvl->solve) {
                    AMGCL_TIC("coarse");
                    (*lvl->solve)(rhs, x);
                    AMGCL_TOC("coarse");
                } else {
                    AMGCL_TIC("relax");
                    for(size_t i = 0; i < prm.npre;  ++i) lvl->relax->apply_pre(*A, rhs, x, *lvl->t);
                    for(size_t i = 0; i < prm.npost; ++i) lvl->relax->apply_post(*A, rhs, x, *lvl->t);
                    AMGCL_TOC("relax");
                }
            } else {
                for (size_t j = 0; j < prm.ncycle; ++j) {
                    AMGCL_TIC("relax");
                    for(size_t i = 0; i < prm.npre; ++i)
                        lvl->relax->apply_pre(*A, rhs, x, *lvl->t);
                    AMGCL_TOC("relax");

                    backend::residual(rhs, *A, x, *lvl->t);

                    backend::spmv(math::identity<scalar_type>(), *lvl->R, *lvl->t, math::zero<scalar_type>(), *nxt->f);

                    backend::clear(*nxt->u);
                    cycle(nxt->A, nxt, *nxt->f, *nxt->u);

                    backend::spmv(math::identity<scalar_type>(), *lvl->P, *nxt->u, math::identity<scalar_type>(), x);

                    AMGCL_TIC("relax");
                    for(size_t i = 0; i < prm.npost; ++i)
                        lvl->relax->apply_post(*A, rhs, x, *lvl->t);
                    AMGCL_TOC("relax");
                }
            }
        }

    template <class B, template <class> class C, template <class> class R>
    friend std::ostream& operator<<(std::ostream &os, const amg<B, C, R> &a);
};

/// Sends information about the AMG hierarchy to output stream.
template <class B, template <class> class C, template <class> class R>
std::ostream& operator<<(std::ostream &os, const amg<B, C, R> &a)
{
    typedef typename amg<B, C, R>::level level;
    std::ios_base::fmtflags ff(os.flags());
    auto fp = os.precision();

    size_t sum_dof = 0;
    size_t sum_nnz = 0;
    size_t sum_mem = 0;

    for(const level &lvl : a.levels) {
        sum_dof += lvl.rows();
        sum_nnz += lvl.nonzeros();
        sum_mem += lvl.bytes();
    }

    os << "Number of levels:    "   << a.levels.size()
        << "\nOperator complexity: " << std::fixed << std::setprecision(2)
        << 1.0 * sum_nnz / a.levels.front().nonzeros()
        << "\nGrid complexity:     " << std::fixed << std::setprecision(2)
        << 1.0 * sum_dof / a.levels.front().rows()
        << "\nMemory footprint:    " << human_readable_memory(sum_mem)
        << "\n\n"
           "level     unknowns       nonzeros      memory\n"
           "---------------------------------------------\n";

    size_t depth = 0;
    for(const level &lvl : a.levels) {
        os << std::setw(5)  << depth++
            << std::setw(13) << lvl.rows()
            << std::setw(15) << lvl.nonzeros()
            << std::setw(12) << human_readable_memory(lvl.bytes())
            << " (" << std::setw(5) << std::fixed << std::setprecision(2)
            << 100.0 * lvl.nonzeros() / sum_nnz
            << "%)" << std::endl;
    }

    os.flags(ff);
    os.precision(fp);
    return os;
}

} // namespace amgcl

#endif
