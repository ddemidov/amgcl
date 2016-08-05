#ifndef AMGCL_SOLVER_LGMRES_HPP
#define AMGCL_SOLVER_LGMRES_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   lgmres.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  LGMRES method.
 *
 * Ported from scipy lgmres. The original code came with the following license:
 * \verbatim
   Copyright (c) 2001, 2002 Enthought, Inc.
   All rights reserved.

   Copyright (c) 2003-2016 SciPy Developers.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

     a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
     b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
     c. Neither the name of Enthought nor the names of the SciPy Developers
        may be used to endorse or promote products derived from this software
        without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
   BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
   OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
   THE POSSIBILITY OF SUCH DAMAGE.
 * \endverbatim
 */

#include <vector>
#include <cmath>

#include <boost/multi_array.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <amgcl/backend/interface.hpp>
#include <amgcl/solver/detail/default_inner_product.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace solver {

/** "Loose" GMRES.
 * \rst
 * The LGMRES algorithm [BaJM05]_  is designed to avoid some problems
 * in the convergence in restarted GMRES, and often converges in fewer
 * iterations.
 * \endrst
 */
template <
    class Backend,
    class InnerProduct = detail::default_inner_product
    >
class lgmres {
    public:
        typedef Backend backend_type;

        typedef typename Backend::vector     vector;
        typedef typename Backend::value_type value_type;
        typedef typename Backend::params     backend_params;

        typedef typename math::scalar_of<value_type>::type scalar_type;
        typedef typename math::rhs_of<value_type>::type rhs_type;
        typedef typename math::inner_product_impl<rhs_type>::return_type coef_type;

        /// Solver parameters.
        struct params {
            /// Number of inner GMRES iterations per each outer iteration.
            int M;

            /// Number of vectors to carry between inner GMRES iterations.
            /**
             * According to [BaJM05], good values are in the range of 1...3.
             * However, note that if you want to use the additional vectors to
             * accelerate solving multiple similar problems, larger values may
             * be beneficial.
             */
            int K;

            /// Whether LGMRES should store also A*v in addition to vectors `v`.
            bool store_Av;

            /// Maximum number of iterations.
            size_t maxiter;

            /// Target residual error.
            scalar_type tol;

            params()
                : M(30), K(3), store_Av(true), maxiter(100), tol(1e-8)
            { }

            params(const boost::property_tree::ptree &p)
                : AMGCL_PARAMS_IMPORT_VALUE(p, M),
                  AMGCL_PARAMS_IMPORT_VALUE(p, K),
                  AMGCL_PARAMS_IMPORT_VALUE(p, store_Av),
                  AMGCL_PARAMS_IMPORT_VALUE(p, maxiter),
                  AMGCL_PARAMS_IMPORT_VALUE(p, tol)
            {
                AMGCL_PARAMS_CHECK(p, (M)(maxiter)(tol));
            }

            void get(boost::property_tree::ptree &p, const std::string &path) const {
                AMGCL_PARAMS_EXPORT_VALUE(p, path, M);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, K);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, store_Av);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, maxiter);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, tol);
            }
        } prm;

        /// Preallocates necessary data structures for the system of size \p n.
        lgmres(
                size_t n,
                const params &prm = params(),
                const backend_params &bprm = backend_params(),
                const InnerProduct &inner_product = InnerProduct()
             )
            : prm(prm), n(n), inner_product(inner_product),
              H(boost::extents[prm.M + prm.K + 1][prm.M + prm.K + 1], boost::fortran_storage_order()),
              y(prm.M + prm.K),
              Ry(prm.M + prm.K),
              q(prm.M + prm.K + 1),
              r(Backend::create_vector(n, bprm)),
              ws(prm.M + prm.K)
        {
            r = Backend::create_vector(n, bprm);

            outer_v_data.reserve(prm.K);
            for(int i = 0; i < prm.K; ++i)
                outer_v_data.push_back(Backend::create_vector(n, bprm));

            if (prm.store_Av) {
                outer_Av_data.reserve(prm.K);
                for(int i = 0; i < prm.K; ++i)
                    outer_Av_data.push_back(Backend::create_vector(n, bprm));
            }

            vs.reserve(prm.M + prm.K);
            for(int i = 0; i < prm.M + prm.K; ++i)
                vs.push_back(Backend::create_vector(n, bprm));
        }

        /* Computes the solution for the given system matrix \p A and the
         * right-hand side \p rhs.  Returns the number of iterations made and
         * the achieved residual as a ``boost::tuple``. The solution vector
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
        boost::tuple<size_t, scalar_type> operator()(
                Matrix  const &A,
                Precond const &P,
                Vec1    const &rhs,
                Vec2          &x
                ) const
        {
            scalar_type rhs_norm = norm(rhs);
            if (rhs_norm < amgcl::detail::eps<scalar_type>(n)) {
                backend::clear(x);
                return boost::make_tuple(0, rhs_norm);
            }

            scalar_type r_norm = math::zero<scalar_type>(), inner_res_0;
            boost::circular_buffer< boost::shared_ptr<vector> > outer_v(prm.K);
            boost::circular_buffer< boost::shared_ptr<vector> > outer_Av(prm.K);

            int H_cols = prm.M + prm.K;
            size_t iter = 0, n_outer = 0;
            for(; iter < prm.maxiter; ) {
                backend::residual(rhs, A, x, *r);

                // Check stopping condition
                r_norm = norm(*r);
                if (r_norm < prm.tol * rhs_norm) break;

                // Inner LGMRES iteration
                P.apply(*r, *vs[0]);
                inner_res_0 = norm(*vs[0]);

                precondition(!math::is_zero(inner_res_0),
                        "Preconditioner returned a zero vector");

                backend::axpby(math::inverse(inner_res_0), *vs[0],
                        math::zero<scalar_type>(), *vs[0]);

                // H is stored in QR factorized form
                amgcl::detail::QR<coef_type, amgcl::detail::col_major> qr;
                for(int j = 0; j <= H_cols; ++j)
                    for(int i = 0; i <= H_cols; ++i)
                        H[i][j] = math::zero<coef_type>();

                qr.compute(H_cols+1, 0, H.data(), H_cols+1);

                const scalar_type eps = std::numeric_limits<scalar_type>::epsilon();
                bool breakdown = false;

                size_t j = 0;
                for(; iter < prm.maxiter && j < prm.M + outer_v.size(); ++j, ++iter) {
                    // -- Arnoldi process:
                    //
                    // Build an orthonormal basis V and matrices W and H such that
                    //     A W = V H
                    // Columns of W, V, and H are stored in `ws`, `vs` and `hs`.
                    //
                    // The first column of V is always the residual vector,
                    // `vs0`; V has *one more column* than the other of the
                    // three matrices.
                    //
                    // The other columns in V are built by feeding in, one by
                    // one, some vectors `z` and orthonormalizing them against
                    // the basis so far. The trick here is to feed in first
                    // some augmentation vectors, before starting to construct
                    // the Krylov basis on `v0`.
                    //
                    // It was shown in [BaJM05] that a good choice (the LGMRES
                    // choice) for these augmentation vectors are the `dx`
                    // vectors obtained from a couple of the previous restart
                    // cycles.
                    //
                    // Note especially that while `vs0` is always the first
                    // column in V, there is no reason why it should also be
                    // the first column in W. (In fact, below `vs0` comes in W
                    // only after the augmentation vectors.)
                    //
                    // The rest of the algorithm then goes as in GMRES, one
                    // solves a minimization problem in the smaller subspace
                    // spanned by W (range) and V (image).

                    boost::shared_ptr<vector> v_new = vs[j+1];
                    boost::shared_ptr<vector> z;

                    if (j < outer_v.size()) {
                        z = outer_v[j];
                    } else if (j == outer_v.size()) {
                        z = vs[0];
                    } else {
                        z = vs[j];
                    }

                    if (j < outer_Av.size()) {
                        backend::copy(*outer_Av[j], *v_new);
                    } else {
                        backend::spmv(math::identity<scalar_type>(), A, *z, math::zero<scalar_type>(), *r);
                        P.apply(*r, *v_new);
                    }

                    scalar_type v_new_norm = norm(*v_new);
                    coef_type   alpha = math::zero<coef_type>();

                    for(size_t i = 0; i <= j; ++i) {
                        H[i][j] = alpha = inner_product(*vs[i], *v_new);
                        backend::axpby(-alpha, *vs[i], math::identity<coef_type>(), *v_new);
                    }
                    H[j+1][j] = norm(*v_new);

                    // Careful with denormals:
                    alpha = math::inverse(H[j+1][j]);
                    if (boost::math::isfinite(alpha))
                        backend::axpby(alpha, *v_new, math::zero<coef_type>(), *v_new);

                    if (!(math::norm(H[j+1][j]) > eps * v_new_norm)) {
                        // v_new essentially in the span of previous vectors,
                        // or we have nans. Bail out after updating the QR
                        // solution.
                        breakdown = true;
                    }

                    ws[j] = z;


                    // -- GMRES optimization problem
                    //
                    // Add new column to H = Q*R
                    qr.append_cols(1);

                    // Transformed least squares problem
                    // || Q R y - inner_res_0 * e_1 ||_2 = min!
                    // Since R = [R'; 0], solution is y = inner_res_0 (R')^{-1} (Q^H)[:j,0]
                    //
                    // Residual is immediately known
                    qr.compute_q(j+2);
                    scalar_type inner_res = std::abs(qr.Q(0,j+1)) * inner_res_0;

                    // Check for termination
                    if (inner_res <= prm.tol * inner_res_0 || breakdown) {
                        ++j;
                        break;
                    }
                }

                precondition(boost::math::isfinite(qr.R(j-1,j-1)), "NaNs encountered in LGMRES");

                // The problem is triangular, but the condition number may be
                // bad (or in case of breakdown the last diagonal entry may be
                // zero), so use lstsq instead of triangular solve.
                //
                // TODO: This is triangular solve for now.
                for(size_t i = 0; i < j; ++i) y[i] = math::adjoint(qr.Q(0, i));
                for(size_t i = j; i --> 0; ) {
                    coef_type rii = qr.R(i,i);
                    if (math::is_zero(rii)) continue;
                    y[i] = math::inverse(rii) * y[i];
                    for(size_t k = 0; k < i; ++k)
                        y[k] -= qr.R(k, i) * y[i];

                    y[i] *= inner_res_0;
                    precondition(boost::math::isfinite(y[i]), "NaNs encountered in LGMRES");
                }

                // -- GMRES terminated: eval solution
                boost::shared_ptr<vector> dx = r;
                sum(j, y, ws, *dx);

                // -- Apply step
                backend::axpby(math::identity<scalar_type>(), *dx, math::identity<scalar_type>(), x);

                // -- Store LGMRES augmented vectors
                scalar_type nx = norm(*dx);

                if(!math::is_zero(nx)) {
                    size_t outer_slot = n_outer % prm.K;
                    ++n_outer;

                    nx = math::inverse(nx);
                    backend::axpby(nx, *dx, math::zero<scalar_type>(), *outer_v_data[outer_slot]);
                    outer_v.push_back(outer_v_data[outer_slot]);

                    if (prm.store_Av) {
                        // q = Q * R * y
                        for(size_t k = 0; k < j; ++k) {
                            coef_type sum = math::zero<coef_type>();
                            for(size_t i = k; i < j; ++i)
                                sum += qr.R(k,i) * y[i];
                            Ry[k] = sum;
                        }

                        for(size_t k = 0; k <= j; ++k) {
                            coef_type sum = math::zero<coef_type>();
                            for(size_t i = 0; i < j; ++i)
                                sum += qr.Q(k,i) * Ry[i];
                            q[k] = sum;
                        }

                        boost::shared_ptr<vector> ax = outer_Av_data[outer_slot];
                        sum(j+1, q, vs, *ax);

                        backend::axpby(nx, *ax, math::zero<scalar_type>(), *ax);
                        outer_Av.push_back(ax);
                    }
                }
            }

            return boost::make_tuple(iter, r_norm / rhs_norm);
        }

        /* Computes the solution for the given right-hand side \p rhs. The
         * system matrix is the same that was used for the setup of the
         * preconditioner \p P.  Returns the number of iterations made and the
         * achieved residual as a ``boost::tuple``. The solution vector \p x
         * provides initial approximation in input and holds the computed
         * solution on output.
         */
        template <class Precond, class Vec1, class Vec2>
        boost::tuple<size_t, scalar_type> operator()(
                Precond const &P,
                Vec1    const &rhs,
                Vec2          &x
                ) const
        {
            return (*this)(P.system_matrix(), P, rhs, x);
        }

    private:
        size_t n;
        InnerProduct inner_product;

        mutable boost::multi_array<coef_type, 2> H;
        mutable std::vector<coef_type> y, Ry, q;
        mutable boost::shared_ptr<vector> r;
        mutable std::vector< boost::shared_ptr<vector> > outer_v_data, outer_Av_data;
        mutable std::vector< boost::shared_ptr<vector> > vs, ws;



        template <class Vec>
        scalar_type norm(const Vec &x) const {
            return std::abs(sqrt(inner_product(x, x)));
        }

        // x = sum(c[i] * v[i])
        static void sum(
                size_t n,
                const std::vector<coef_type> &c,
                const std::vector< boost::shared_ptr<vector> > &v,
                vector &x
                )
        {
            size_t i = 0;

            if (n < 2) {
                i = 1;
                backend::axpby(c[0], *v[0], math::zero<coef_type>(), x);
            } else {
                i = 2;
                backend::axpbypcz(c[0], *v[0], c[1], *v[1], math::zero<coef_type>(), x);
            }

            for(; i + 1 < n; i += 2) {
                backend::axpbypcz(c[i], *v[i], c[i+1], *v[i+1], math::identity<coef_type>(), x);
            }

            for(; i < n; ++i) {
                backend::axpby(c[i], *v[i], math::identity<coef_type>(), x);
            }
        }
};

} // namespace solver
} // namespace amgcl

#endif
