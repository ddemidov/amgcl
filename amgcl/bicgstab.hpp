#ifndef AMGCL_BICGSTAB_HPP
#define AMGCL_BICGSTAB_HPP

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
 * \file   bicgstab.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Stabilized BiConjugate Gradient method.
 *
 * Implementation is based on \ref Templates_1994 "Barrett (1994)"
 */

#include <tuple>
#include <stdexcept>

namespace amgcl {

/// Controlling parameters.
struct bicg_tag {
    int maxiter; ///< Maximum number of iterations.
    double tol;  ///< The desired precision.

    bicg_tag(int maxiter = 100, double tol = 1e-8)
        : maxiter(maxiter), tol(tol)
    {}
};

/// Stabilized BiConjugate Gradient method.
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
 * \returns a tuple containing number of iterations made and precision
 * achieved.
 */
template <class matrix, class vector, class precond>
std::tuple< int, typename value_type<vector>::type >
solve(const matrix &A, const vector &rhs, const precond &P, vector &x, bicg_tag prm = bicg_tag())
{
    typedef typename value_type<vector>::type value_t;

    const auto n = x.size();

    vector r (n);
    vector p (n);
    vector v (n);
    vector s (n);
    vector t (n);
    vector rh(n);
    vector ph(n);
    vector sh(n);

    rh = r = rhs - A * x;

    value_t rho1  = 0, rho2  = 0;
    value_t alpha = 0, omega = 0;

    value_t norm_of_rhs = norm(rhs);

    int     iter;
    value_t res = 2 * prm.tol;
    for(iter = 0; res > prm.tol && iter < prm.maxiter; ++iter) {
        rho2 = rho1;
        rho1 = inner_prod(r, rh);

        if (fabs(rho1) < 1e-32)
            throw std::logic_error("Zero rho in BiCGStab");

        if (iter)
            p = r + ((rho1 * alpha) / (rho2 * omega)) * (p - omega * v);
        else
            p = r;

        clear(ph);
        P.apply(p, ph);

        v = A * ph;

        alpha = rho1 / inner_prod(rh, v);

        s = r - alpha * v;

        if ((res = norm(s) / norm_of_rhs) < prm.tol) {
            x += alpha * ph;
        } else {
            clear(sh);
            P.apply(s, sh);

            t = A * sh;

            omega = inner_prod(t, s) / inner_prod(t, t);

            if (fabs(omega) < 1e-32)
                throw std::logic_error("Zero omega in BiCGStab");

            x += alpha * ph + omega * sh;
            r = s - omega * t;

            res = norm(r) / norm_of_rhs;
        }
    }

    return std::make_tuple(iter, res);
}

} // namespace amgcl

#endif
