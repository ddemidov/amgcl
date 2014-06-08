#ifndef AMGCL_SOLVERS_BICGSTAB_HPP
#define AMGCL_SOLVERS_BICGSTAB_HPP

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
 * \file   amgcl/solver/bicgstab.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  BiCGStab iterative method.
 */

#include <boost/tuple/tuple.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace solver {

/// BiCGStab iterative solver.
/**
 * \param Backend Backend for temporary structures allocation.
 * \ingroup solvers
 * \sa \cite Barrett1994
 */
template <class Backend>
class bicgstab {
    public:
        typedef typename Backend::vector     vector;
        typedef typename Backend::value_type value_type;
        typedef typename Backend::params     backend_params;

        /// Solver parameters.
        struct params {
            /// Maximum number of iterations.
            size_t maxiter;

            /// Target residual error.
            value_type tol;

            params(size_t maxiter = 100, value_type tol = 1e-8)
                : maxiter(maxiter), tol(tol)
            {}
        };

        /// \copydoc amgcl::solver::cg::cg
        bicgstab(
                size_t n,
                const params &prm = params(),
                const backend_params &backend_prm = backend_params()
                )
            : prm(prm), n(n),
              r ( Backend::create_vector(n, backend_prm) ),
              p ( Backend::create_vector(n, backend_prm) ),
              v ( Backend::create_vector(n, backend_prm) ),
              s ( Backend::create_vector(n, backend_prm) ),
              t ( Backend::create_vector(n, backend_prm) ),
              rh( Backend::create_vector(n, backend_prm) ),
              ph( Backend::create_vector(n, backend_prm) ),
              sh( Backend::create_vector(n, backend_prm) )
        { }

        /// Solves the linear system for the given system matrix.
        /**
         * \param A   System matrix.
         * \param P   Preconditioner.
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
        template <class Matrix, class Precond>
        boost::tuple<size_t, value_type> operator()(
                Matrix  const &A,
                Precond const &P,
                vector  const &rhs,
                vector        &x
                ) const
        {
            backend::residual(rhs, A, x, *r);
            backend::copy(*r, *rh);

            value_type rho1  = 0, rho2  = 0;
            value_type alpha = 0, omega = 0;

            value_type norm_of_rhs = backend::norm(rhs);

            size_t     iter;
            value_type res = 2 * prm.tol;

            for(iter = 0; res > prm.tol && iter < prm.maxiter; ++iter) {
                rho2 = rho1;
                rho1 = backend::inner_product(*r, *rh);

                precondition(rho1, "Zero rho in BiCGStab");

                if (iter) {
                    value_type beta = (rho1 * alpha) / (rho2 * omega);
                    backend::axpbypcz(1, *r, -beta * omega, *v, beta, *p);
                } else {
                    backend::copy(*r, *p);
                }

                P(*p, *ph);

                backend::spmv(1, A, *ph, 0, *v);

                alpha = rho1 / backend::inner_product(*rh, *v);

                backend::axpbypcz(1, *r, -alpha, *v, 0, *s);

                if ((res = backend::norm(*s) / norm_of_rhs) < prm.tol) {
                    backend::axpby(alpha, *ph, 1, x);
                } else {
                    P(*s, *sh);

                    backend::spmv(1, A, *sh, 0, *t);

                    omega = backend::inner_product(*t, *s)
                          / backend::inner_product(*t, *t);

                    precondition(omega, "Zero omega in BiCGStab");

                    backend::axpbypcz(alpha, *ph, omega, *sh, 1, x);
                    backend::axpbypcz(1, *s, -omega, *t, 0, *r);

                    res = backend::norm(*r) / norm_of_rhs;
                }
            }

            return boost::make_tuple(iter, res);
        }

        /// Solves the linear system for the same matrix that was used for the AMG preconditioner construction.
        /**
         * \param P   AMG preconditioner.
         * \param rhs Right-hand side.
         * \param x   Solution vector.
         */
        template <class Precond>
        boost::tuple<size_t, value_type> operator()(
                Precond const &P,
                vector  const &rhs,
                vector        &x
                ) const
        {
            return (*this)(P.top_matrix(), P, rhs, x);
        }


    private:
        params prm;
        size_t n;

        boost::shared_ptr<vector> r;
        boost::shared_ptr<vector> p;
        boost::shared_ptr<vector> v;
        boost::shared_ptr<vector> s;
        boost::shared_ptr<vector> t;
        boost::shared_ptr<vector> rh;
        boost::shared_ptr<vector> ph;
        boost::shared_ptr<vector> sh;
};

} // namespace solver
} // namespace amgcl


#endif
