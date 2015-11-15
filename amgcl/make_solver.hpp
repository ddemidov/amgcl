#ifndef AMGCL_MAKE_SOLVER_HPP
#define AMGCL_MAKE_SOLVER_HPP

/*
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/make_solver.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Tie an iterative solver and a preconditioner in a single class.
 */

#include <boost/type_traits.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {

/// Convenience class that wraps a preconditioner and an iterative solver
template <
    class Precond,
    class IterativeSolver
    >
class make_solver {
    public:
        typedef typename Precond::backend_type Backend;

        BOOST_STATIC_ASSERT_MSG(
                (boost::is_same<Backend, typename IterativeSolver::backend_type>::value),
                "Backends for preconditioner and iterative solver should coinside"
                );

        typedef typename Backend::value_type value_type;
        typedef typename Backend::params backend_params;
        typedef typename backend::builtin<value_type>::matrix build_matrix;

        struct params {
            typename Precond::params         precond;
            typename IterativeSolver::params solver;

            params() {}

            params(const boost::property_tree::ptree &p)
                : AMGCL_PARAMS_IMPORT_CHILD(p, precond),
                  AMGCL_PARAMS_IMPORT_CHILD(p, solver)
            {}

            void get(
                    boost::property_tree::ptree &p,
                    const std::string &path = ""
                    ) const
            {
                AMGCL_PARAMS_EXPORT_CHILD(p, path, precond);
                AMGCL_PARAMS_EXPORT_CHILD(p, path, solver);
            }
        } prm;

        /// Constructs the preconditioner and creates iterative solver.
        template <class Matrix>
        make_solver(
                const Matrix &A,
                const params &prm = params(),
                const backend_params &bprm = backend_params()
                ) :
            prm(prm), n(backend::rows(A)),
            P(A, prm.precond, bprm),
            S(backend::rows(A), prm.solver, bprm)
        {}

        /// Constructs the preconditioner and creates iterative solver.
        make_solver(
                boost::shared_ptr<build_matrix> A,
                const params &prm = params(),
                const backend_params &bprm = backend_params()
                ) :
            prm(prm), n(backend::rows(*A)),
            P(A, prm.precond, bprm),
            S(backend::rows(*A), prm.solver, bprm)
        {}

        /// Solves the linear system for the given system matrix.
        /**
         * \param A   System matrix.
         * \param rhs Right-hand side.
         * \param x   Solution vector.
         *
         * The system matrix may differ from the matrix used for the AMG
         * preconditioner construction. This may be used for the solution of
         * non-stationary problems with slowly changing coefficients. There is
         * a strong chance that AMG built for one time step will act as a
         * reasonably good preconditioner for several subsequent time steps
         * \cite Demidov2012.
         */
        template <class Matrix, class Vec1, class Vec2>
        boost::tuple<size_t, value_type> operator()(
                Matrix  const &A,
                Vec1    const &rhs,
#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
                Vec2          &x
#else
                Vec2          &&x
#endif
                ) const
        {
            return S(A, P, rhs, x);
        }

        /// Solves the linear system for the given right-hand side.
        /**
         * \param rhs Right-hand side.
         * \param x   Solution vector.
         */
        template <class Vec1, class Vec2>
        boost::tuple<size_t, value_type> operator()(
                Vec1    const &rhs,
#ifdef BOOST_NO_CXX11_RVALUE_REFERENCES
                Vec2          &x
#else
                Vec2          &&x
#endif
                ) const
        {
            return S(P, rhs, x);
        }

        /// Acts as a preconditioner.
        /**
         * \param rhs Right-hand side.
         * \param x   Solution vector.
         */
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
            backend::clear(x);
            (*this)(rhs, x);
        }

        /// Reference to the constructed AMG hierarchy.
        const Precond& precond() const {
            return P;
        }

        /// Reference to the iterative solver.
        const IterativeSolver& solver() const {
            return S;
        }

        /// The system matrix in the backend format.
        typename Precond::matrix const& system_matrix() const {
            return P.system_matrix();
        }

        /// Fills the property tree with the actual parameters used.
        void get_params(boost::property_tree::ptree &p) const {
            prm.get(p);
        }

        /// Size of the system matrix.
        size_t size() const {
            return n;
        }

    private:
        size_t           n;
        Precond          P;
        IterativeSolver  S;
};

} // namespace amgcl

#endif
