#ifndef AMGCL_GMRES_HPP
#define AMGCL_GMRES_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  GMRES method.
 *
 * Implementation is based on \ref Templates_1994 "Barrett (1994)"
 */

#include <vector>
#include <utility>
#include <algorithm>

#include <amgcl/common.hpp>

namespace amgcl {

template <class Vector>
struct gmres_data {
    typedef typename value_type<Vector>::type value_type;

    int M;
    std::vector<value_type> H, s, cs, sn, y;
    Vector r, w;
    std::vector<Vector> v;

    gmres_data(int M, size_t n)
        : M(M), H(M * (M + 1)), s(M + 1), cs(M + 1), sn(M + 1), y(M + 1),
          r(n), w(n), v(M + 1)
    {
        for(BOOST_AUTO(vp, v.begin()); vp != v.end(); ++vp) vp->resize(n);
    }

    static void apply_plane_rotation(value_type &dx, value_type &dy, value_type cs, value_type sn) {
        value_type tmp = cs * dx + sn * dy;
        dy = -sn * dx + cs * dy;
        dx = tmp;
    }

    static void generate_plane_rotation(value_type dx, value_type dy, value_type &cs, value_type &sn) {
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

    void update(Vector &x, int k) {
        std::copy(s.begin(), s.end(), y.begin());

        for (int i = k; i >= 0; --i) {
            y[i] /= H[i * M + i];
            for (int j = i - 1; j >= 0; --j)
                y[j] -= H[j * M + i] * y[i];
        }

        for (int j = 0; j <= k; j++)
            x += y[j] * v[j];
    }

    template <class matrix, class precond>
    value_type restart(
            const matrix &A, const Vector &rhs, const precond &P, const Vector &x
            )
    {
        residual(A, x, rhs, w);
        clear(r);
        P.apply(w, r);

        s[0] = norm(r);
        v[0] = r / s[0];

        std::fill(s.begin() + 1, s.end(), static_cast<value_type>(0));

        return s[0];
    }

    template <class matrix, class precond>
    value_type iteration(const matrix &A, const precond &P, int i) {
        axpy(A, v[i], r);
        clear(w);
        P.apply(r, w);

        for(int k = 0; k <= i; ++k) {
            H[k * M + i] = inner_prod(w, v[k]);
            w -= H[k * M + i] * v[k];
        }

        H[(i+1) * M + i] = norm(w);

        v[i+1] = w / H[(i+1) * M + i];

        for(int k = 0; k < i; ++k)
            apply_plane_rotation(H[k * M + i], H[(k+1) * M + i], cs[k], sn[k]);

        generate_plane_rotation(H[i * M + i], H[(i+1) * M + i], cs[i], sn[i]);
        apply_plane_rotation(H[i * M + i], H[(i+1) * M + i], cs[i], sn[i]);
        apply_plane_rotation(s[i], s[i+1], cs[i], sn[i]);

        return fabs(s[i + 1]);
    }
};

/// GMRES method.
/**
 * Implementation is based on \ref Templates_1994 "Barrett (1994)"
 *
 * \param A   The system matrix.
 * \param rhs The right-hand side.
 * \param P   The preconditioner. Should provide apply(rhs, x) method.
 * \param x   The solution. Contains an initial approximation on input, and
 *            the approximated solution on output.
 * \param prm The control parameters.
 *
 * \returns a pair containing number of iterations made and precision
 * achieved.
 *
 * \ingroup iterative
 */
template <class matrix, class vector, class precond>
std::pair< int, typename value_type<vector>::type >
solve(const matrix &A, const vector &rhs, const precond &P, vector &x, gmres_tag prm = gmres_tag())
{
    typedef typename value_type<vector>::type value_t;
    const size_t n = x.size();

    gmres_data<vector> gmres(prm.M, n);

    int     iter = 0;
    value_t res;

    do {
        res = gmres.restart(A, rhs, P, x);

        for(int i = 0; i < prm.M && iter < prm.maxiter; ++i, ++iter) {
            res = gmres.iteration(A, P, i);

	    if (res < prm.tol) {
                gmres.update(x, i);
		return std::make_pair(iter + 1, res);
	    };
	}

        gmres.update(x, prm.M - 1);
    } while (iter < prm.maxiter && res > prm.tol);

    return std::make_pair(iter, res);
}

} // namespace amgcl

#endif
