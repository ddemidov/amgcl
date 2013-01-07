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

#include <utility>
#include <algorithm>

#include <boost/array.hpp>

#include <amgcl/common.hpp>

namespace amgcl {

namespace gmres {

//---------------------------------------------------------------------------
template <typename value_t>
void apply_plane_rotation(value_t &dx, value_t &dy, value_t cs, value_t sn) {
    value_t tmp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = tmp;
}

//---------------------------------------------------------------------------
template <typename value_t>
void generate_plane_rotation(value_t dx, value_t dy, value_t &cs, value_t &sn) {
    if (dy == 0) {
	cs = 1;
	sn = 0;
    } else if (fabs(dy) > fabs(dx)) {
	value_t tmp = dx / dy;
	sn = 1 / sqrt(1 + tmp * tmp);
	cs = tmp * sn;
    } else {
	value_t tmp = dy / dx;
	cs = 1 / sqrt(1 + tmp * tmp);
	sn = tmp * cs;
    }
}

//---------------------------------------------------------------------------
template <size_t M, class vector>
void update(
        vector &x, int k,
        boost::array<boost::array<typename value_type<vector>::type, M>, M + 1> &h,
        const boost::array<typename value_type<vector>::type, M + 1> &s,
        const boost::array<vector, M + 1> &v,
        boost::array<typename value_type<vector>::type, M + 1> &y
        )
{
    std::copy(s.begin(), s.end(), y.begin());

    for (int i = k; i >= 0; --i) {
	y[i] /= h[i][i];
	for (int j = i - 1; j >= 0; --j)
	    y[j] -= h[j][i] * y[i];
    }

    for (int j = 0; j <= k; j++)
        x += y[j] * v[j];
}

} // namespace gmres

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
template <size_t M, class matrix, class vector, class precond>
std::pair< int, typename value_type<vector>::type >
solve(const matrix &A, const vector &rhs, const precond &P, vector &x, gmres_tag<M> prm = gmres_tag<M>())
{
    typedef typename value_type<vector>::type value_t;
    const size_t n = x.size();


    boost::array<boost::array<value_t, M>, M + 1> H;
    boost::array<value_t, M + 1> s, cs, sn, y;

    vector r(n), w(n);
    boost::array<vector, M + 1> v;
    for(BOOST_AUTO(vp, v.begin()); vp != v.end(); ++vp) vp->resize(n);

    w = rhs - A * x;
    clear(r);
    P.apply(w, r);

    value_t res  = norm(r);
    int     iter = 0;

    while(iter < prm.maxiter && res > prm.tol) {
        if (iter > 0) {
            w = rhs - A * x;
            clear(r);
            P.apply(w, r);
            res = norm(r);
        }

        v[0] = r / res;

        std::fill(s.begin(), s.end(), static_cast<value_t>(0));
        s[0] = res;

        for(int i = 0; i < M && iter < prm.maxiter; ++i, ++iter) {
            r = A * v[i];
            clear(w);
            P.apply(r, w);

            for(int k = 0; k <= i; ++k) {
                H[k][i] = inner_prod(w, v[k]);
                w -= H[k][i] * v[k];
            }

            H[i+1][i] = norm(w);

            v[i+1] = w / H[i+1][i];

            for(int k = 0; k < i; ++k)
                gmres::apply_plane_rotation(H[k][i], H[k+1][i], cs[k], sn[k]);

            gmres::generate_plane_rotation(H[i][i], H[i+1][i], cs[i], sn[i]);
            gmres::apply_plane_rotation(H[i][i], H[i+1][i], cs[i], sn[i]);
            gmres::apply_plane_rotation(s[i], s[i+1], cs[i], sn[i]);

	    res = fabs(s[i+1]);

	    if (res < prm.tol) {
                gmres::update(x, i, H, s, v, y);
		return std::make_pair(iter + 1, res);
	    };
	}

        gmres::update(x, M - 1, H, s, v, y);
    }

    return std::make_pair(iter, res);
}

} // namespace amgcl

#endif
