#ifndef AMGCL_SOLVER_GMRES_HPP
#define AMGCL_SOLVER_GMRES_HPP

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
 * \file   gmres.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  GMRES method.
 *
 * Implementation is based on \ref Templates_1994 "Barrett (1994)"
 */

#include <vector>
#include <cmath>

#include <boost/multi_array.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/range/algorithm.hpp>

#include <amgcl/backend/interface.hpp>

namespace amgcl {
namespace solver {

template <class Backend>
class gmres {
    public:
        typedef typename Backend::vector     vector;
        typedef typename Backend::value_type value_type;
        typedef typename Backend::params     backend_params;

        gmres(
                size_t n, const backend_params &prm = backend_params(),
                int M = 50, size_t maxiter = 100, value_type tol = 1e-8
             )
            : M(M), maxiter(maxiter), tol(tol),
              H(boost::extents[M + 1][M]),
              s(M + 1), cs(M + 1), sn(M + 1), y(M + 1),
              r( Backend::create_vector(n, prm) ),
              w( Backend::create_vector(n, prm) )
        {
            v.reserve(M + 1);
            for(int i = 0; i <= M; ++i)
                v.push_back( Backend::create_vector(n, prm) );
        }

        template <class Matrix, class Precond>
        boost::tuple<size_t, value_type> operator()(
                Matrix  const &A,
                Precond const &P,
                vector  const &rhs,
                vector        &x
                )
        {
            size_t     iter = 0;
            value_type res;

            do {
                res = restart(A, rhs, P, x);

                for(int i = 0; i < M && iter < maxiter; ++i, ++iter) {
                    res = iteration(A, P, i);

                    if (res < tol) {
                        update(x, i);
                        return boost::make_tuple(iter + 1, res);
                    };
                }

                update(x, M-1);
            } while (iter < maxiter && res > tol);

            return boost::make_tuple(iter, res);
        }

        template <class Precond>
        boost::tuple<size_t, value_type> operator()(
                Precond const &P,
                vector  const &rhs,
                vector        &x
                )
        {
            return (*this)(P.top_matrix(), P, rhs, x);
        }
    private:
        int        M;
        size_t     maxiter;
        value_type tol;

        boost::multi_array<value_type, 2> H;
        std::vector<value_type> s, cs, sn, y;
        boost::shared_ptr<vector> r, w;
        std::vector< boost::shared_ptr<vector> > v;

        static void apply_plane_rotation(
                value_type &dx, value_type &dy, value_type cs, value_type sn
                )
        {
            value_type tmp = cs * dx + sn * dy;
            dy = -sn * dx + cs * dy;
            dx = tmp;
        }

        static void generate_plane_rotation(
                value_type dx, value_type dy, value_type &cs, value_type &sn
                )
        {
            if (dy == 0) {
                cs = 1;
                sn = 0;
            } else if (fabs(dy) > fabs(dx)) {
                value_type tmp = dx / dy;
                sn = 1 / sqrt(1 + tmp * tmp);
                cs = tmp * sn;
            } else {
                value_type tmp = dy / dx;
                cs = 1 / sqrt(1 + tmp * tmp);
                sn = tmp * cs;
            }
        }

        void update(vector &x, int k) {
            boost::range::copy(s, y.begin());

            for (int i = k; i >= 0; --i) {
                y[i] /= H[i][i];
                for (int j = i - 1; j >= 0; --j)
                    y[j] -= H[j][i] * y[i];
            }

            // Unroll the loop
            int j = 0;
            for (; j <= k; j += 2)
                backend::axpbypcz(y[j], *v[j], y[j+1], *v[j+1], 1, x);
            for (; j <= k; ++j)
                backend::axpby(y[j], *v[j], 1, x);
        }

        template <class Matrix, class Precond>
        value_type restart(const Matrix &A, const vector &rhs,
                const Precond &P, const vector &x)
        {
            backend::residual(rhs, A, x, *w);
            P(*w, *r);

            s[0] = backend::norm(*r);
            backend::axpby(1 / s[0], *r, 0, *v[0]);

            std::fill(s.begin() + 1, s.end(), value_type());

            return s[0];
        }

        template <class Matrix, class Precond>
        value_type iteration(const Matrix &A, const Precond &P, int i)
        {
            backend::spmv(1, A, *v[i], 0, *r);
            P(*r, *w);

            for(int k = 0; k <= i; ++k) {
                H[k][i] = backend::inner_product(*w, *v[k]);
                backend::axpby(-H[k][i], *v[k], 1, *w);
            }

            H[i+1][i] = backend::norm(*w);

            backend::axpby(1 / H[i+1][i], *w, 0, *v[i+1]);

            for(int k = 0; k < i; ++k)
                apply_plane_rotation(H[k][i], H[k+1][i], cs[k], sn[k]);

            generate_plane_rotation(H[i][i], H[i+1][i], cs[i], sn[i]);
            apply_plane_rotation(H[i][i], H[i+1][i], cs[i], sn[i]);
            apply_plane_rotation(s[i], s[i+1], cs[i], sn[i]);

            return fabs(s[i+1]);
        }
};

} // namespace solver
} // namespace amgcl

#endif
