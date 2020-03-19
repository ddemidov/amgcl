#ifndef AMGCL_SOLVER_MINRES_HPP
#define AMGCL_SOLVER_MINRES_HPP

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
 * \file   amgcl/solver/minres.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Minimum Residual iterative solver.
 *
 * Ported from scipy minres. The original code came with the following license:
 * \verbatim
Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endverbatim
 */

#include <tuple>
#include <amgcl/backend/interface.hpp>
#include <amgcl/solver/detail/default_inner_product.hpp>
#include <amgcl/util.hpp>

namespace amgcl {

/// Iterative solvers
namespace solver {

// Minimum Residual iteration
template <
    class Backend,
    class InnerProduct = detail::default_inner_product
    >
class minres {
    public:
        typedef Backend backend_type;

        typedef typename Backend::vector     vector;
        typedef typename Backend::value_type value_type;
        typedef typename Backend::params     backend_params;

        typedef typename math::scalar_of<value_type>::type scalar_type;

        typedef typename math::inner_product_impl<
            typename math::rhs_of<value_type>::type
            >::return_type coef_type;

        /// Solver parameters.
        struct params {
            /// Maximum number of iterations.
            size_t maxiter;

            /// Target relative residual error.
            scalar_type tol;

            /// Target absolute residual error.
            scalar_type abstol;

            params()
                : maxiter(100), tol(1e-8),
                  abstol(std::numeric_limits<scalar_type>::min())
            {}

#ifndef AMGCL_NO_BOOST
            params(const boost::property_tree::ptree &p)
                : AMGCL_PARAMS_IMPORT_VALUE(p, maxiter),
                  AMGCL_PARAMS_IMPORT_VALUE(p, tol),
                  AMGCL_PARAMS_IMPORT_VALUE(p, abstol)
            {
                check_params(p, {"maxiter", "tol", "abstol"});
            }

            void get(boost::property_tree::ptree &p, const std::string &path) const {
                AMGCL_PARAMS_EXPORT_VALUE(p, path, maxiter);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, tol);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, abstol);
            }
#endif
        };

        /// Preallocates necessary data structures for the system of size \p n.
        minres(
                size_t n,
                const params &prm = params(),
                const backend_params &backend_prm = backend_params(),
                const InnerProduct &inner_product = InnerProduct()
          ) : prm(prm), n(n),
              r1(Backend::create_vector(n, backend_prm)),
              r2(Backend::create_vector(n, backend_prm)),
              y(Backend::create_vector(n, backend_prm)),
              v(Backend::create_vector(n, backend_prm)),
              w(Backend::create_vector(n, backend_prm)),
              w1(Backend::create_vector(n, backend_prm)),
              w2(Backend::create_vector(n, backend_prm)),
              inner_product(inner_product)
        { }

        /* Computes the solution for the given system matrix \p A and the
         * right-hand side \p rhs.  Returns the number of iterations made and
         * the achieved residual as a ``std::tuple``. The solution vector
         * \p x provides initial approximation in input and holds the computed
         * solution on output.
         *
         * The system matrix may differ from the matrix used during
         * initialization. This may be used for the solution of non-stationary
         * problems with slowly changing coefficients. There is a strong chance
         * that a preconditioner built for a time step will act as a reasonably
         * good preconditioner for several subsequent time steps [DeSh12]_.
         */
        template <class Matrix, class Precond, class Vec1, class Vec2>
        std::tuple<size_t, scalar_type> operator()(
                const Matrix &A, const Precond &P, const Vec1 &rhs, Vec2 &&x) const
        {
            // TODO: not working for complex value type

            static const coef_type one  = math::identity<coef_type>();
            static const coef_type zero = math::zero<coef_type>();

            scalar_type norm_rhs = norm(rhs);
            if (norm_rhs < amgcl::detail::eps<scalar_type>(1)) {
                backend::clear(x);
                return std::make_tuple(0, norm_rhs);
            }

            // Set up y and v for the first Lanczos vector v1.
            // y  =  beta1 P' v1,  where  P = C**(-1).
            // v is really P' v1.
            backend::residual(rhs, A, x, *r1);
            P.apply(*r1, *y);

            scalar_type beta1 = sqrt(math::norm(inner_product(*r1, *y)));
            if (math::is_zero(beta1)) {
                backend::clear(x);
                return std::make_tuple(0, norm_rhs);
            }

            scalar_type eps = std::numeric_limits<scalar_type>::epsilon();

            scalar_type Anorm  = math::zero<scalar_type>();
            scalar_type Acond  = math::zero<scalar_type>();
            scalar_type rnorm  = math::zero<scalar_type>();
            scalar_type ynorm  = math::zero<scalar_type>();
            scalar_type beta   = beta1;
            scalar_type oldb   = math::zero<scalar_type>();
            scalar_type dbar   = math::zero<scalar_type>();
            scalar_type epsln  = math::zero<scalar_type>();
            scalar_type qrnorm = beta1;
            scalar_type phibar = beta1;
            scalar_type rhs1   = beta1;
            scalar_type rhs2   = math::zero<scalar_type>();
            scalar_type tnorm2 = math::zero<scalar_type>();
            scalar_type gmax   = math::zero<scalar_type>();
            scalar_type gmin   = std::numeric_limits<scalar_type>::max();
            coef_type   cs     = math::constant<coef_type>(-1);
            coef_type   sn     = math::zero<coef_type>();

            scalar_type res_norm = beta1;

            backend::clear(*w);
            backend::clear(*w2);
            backend::copy(*r1, *r2);

            size_t iter = 0;
            for(int stop = false; iter < prm.maxiter;) {
                backend::axpby(math::inverse(beta), *y, zero, *v);
                backend::spmv(one, A, *v, zero, *y);

                if (iter) {
                    backend::axpby(-beta / oldb, *r1, one, *y);
                }

                coef_type alfa = inner_product(*v, *y);
                backend::axpby(-alfa / beta, *r2, one, *y);
                std::swap(r1, r2);
                backend::copy(*y, *r2);
                P.apply(*r2, *y);

                oldb = beta;
                beta = sqrt(math::norm(inner_product(*r2, *y)));
                tnorm2 += math::norm(alfa * alfa) + oldb * oldb + beta * beta;

                if (iter == 0 && beta / beta1 <= 10 * eps) {
                    // beta2 = 0. If M = I, b and x are eigenvectors
                    stop = true; // Terminate later
                }

                // Apply previous rotation Qk-1 to get
                //   [deltak epslnk+1] = [cs  sn][dbark    0   ]
                //   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
                auto oldeps = epsln;
                auto delta = cs * dbar + sn * alfa;
                auto gbar  = sn * dbar - cs * alfa;

                epsln =  sn * beta;
                dbar  = -cs * beta;

                auto root = sqrt(gbar * gbar + dbar * dbar);
                Anorm = phibar * root;

                // Compute the next plane rotation Qk
                auto gamma = std::max(sqrt(gbar * gbar + beta * beta), eps);
                auto igamm = math::inverse(gamma);

                cs = gbar * igamm;
                sn = beta * igamm;

                auto phi = cs * phibar;
                phibar   = sn * phibar;

                // Update  x
                std::swap(w1, w2);
                backend::copy(*w, *w2);
                backend::axpbypcz(-oldeps*igamm, *w1, -delta*igamm, *w2, zero, *w);
                backend::axpby(igamm, *v, one, *w);
                backend::axpby(phi, *w, one, x);

                // Go round again
                gmax = std::max(gmax, gamma);
                gmin = std::min(gmin, gamma);

                auto z = rhs1 * igamm;
                rhs1   = rhs2 - delta * z;
                rhs2   = -epsln * z;

                // Estimate various norms and test for convergence
                Anorm = sqrt(tnorm2);
                ynorm = norm(x);
                auto epsx = Anorm * ynorm * eps;

                qrnorm = phibar;
                rnorm  = qrnorm;

                scalar_type test1, test2;
                if (math::is_zero(ynorm) || math::is_zero(Anorm))
                    test1 = std::numeric_limits<scalar_type>::infinity();
                else
                    test1 = rnorm / (Anorm * ynorm);

                if (math::is_zero(Anorm))
                    test2 = std::numeric_limits<scalar_type>::infinity();
                else
                    test2 = root / Anorm;

                res_norm = test1;

                // Estimate  cond(A).
                // In this version we look at the diagonals of  R  in the
                // factorization of the lower Hessenberg matrix,  Q * H = R,
                // where H is the tridiagonal matrix from Lanczos with one
                // extra row, beta(k+1) e_k^T.
                Acond = gmax / gmin;

                // See if any of the stopping criteria are satisfied.
                // In rare cases, stop is already set above (Abar = const*I).
                ++iter;
                if (stop) break;

                auto t1 = 1 + test1; // These tests work if tol < eps
                auto t2 = 1 + test2;

                // A least-squares solution was found, given rtol
                if (t2 <= 1 || test2 <= prm.tol) break;

                // A solution to Ax = b was found, given rtol
                if (t1 <= 1 || test1 <= prm.tol) break;

                // x has converged to an eigenvector
                if (Acond >= 0.1/eps) break;

                // Reasonable accuracy achieved, given eps
                if (epsx >= beta1) break;
            }

            // Compute true residual
            backend::residual(rhs, A, x, *r1);
            res_norm = norm(*r1);
            return std::make_tuple(iter, res_norm / norm_rhs);
        }

        /* Computes the solution for the given right-hand side \p rhs. The
         * system matrix is the same that was used for the setup of the
         * preconditioner \p P.  Returns the number of iterations made and the
         * achieved residual as a ``std::tuple``. The solution vector \p x
         * provides initial approximation in input and holds the computed
         * solution on output.
         */
        template <class Precond, class Vec1, class Vec2>
        std::tuple<size_t, scalar_type> operator()(
                const Precond &P, const Vec1 &rhs, Vec2 &&x) const
        {
            return (*this)(P.system_matrix(), P, rhs, x);
        }

        size_t bytes() const {
            return backend::bytes(*r1)
                 + backend::bytes(*r2)
                 + backend::bytes(*y) 
                 + backend::bytes(*v)
                 + backend::bytes(*w)
                 + backend::bytes(*w1)
                 + backend::bytes(*w2);
        }

        friend std::ostream& operator<<(std::ostream &os, const minres &s) {
            return os
                << "Type:             MINRES"
                << "\nUnknowns:         " << s.n
                << "\nMemory footprint: " << human_readable_memory(s.bytes())
                << std::endl;
        }
    public:
        params prm;

    private:
        size_t n;

        mutable std::shared_ptr<vector> r1, r2, y, v, w, w1, w2;

        InnerProduct inner_product;

        template <class Vec>
        scalar_type norm(const Vec &x) const {
            return sqrt(math::norm(inner_product(x, x)));
        }
};

} // namespace solver
} // namespace amgcl



#endif
