#ifndef AMGCL_SOLVERS_CG_HPP
#define AMGCL_SOLVERS_CG_HPP

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
 * \file   amgcl/solver/cg.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Conjugate Gradient method.
 *
 * Implementation is based on \ref Templates_1994 "Barrett (1994)"
 */

#include <boost/tuple/tuple.hpp>
#include <amgcl/backend/interface.hpp>

namespace amgcl {
namespace solver {

template <class Backend>
class cg {
    public:
        typedef typename Backend::vector     vector;
        typedef typename Backend::value_type value_type;
        typedef typename Backend::params     backend_params;

        cg(size_t n, const backend_params &prm = backend_params(),
                size_t maxiter = 100, value_type tol = 1e-8)
            : n(n), maxiter(maxiter), tol(tol),
              r(Backend::create_vector(n, prm)),
              s(Backend::create_vector(n, prm)),
              p(Backend::create_vector(n, prm)),
              q(Backend::create_vector(n, prm))
        { }

        template <class Matrix, class Precond>
        boost::tuple<size_t, value_type> operator()(
                Matrix  const &A,
                vector  const &rhs,
                Precond const &P,
                vector        &x
                ) const
        {
            backend::residual(rhs, A, x, *r);

            value_type rho1 = 0, rho2 = 0;
            value_type norm_of_rhs = backend::norm(rhs);

            if (norm_of_rhs == 0) {
                backend::clear(x);
                return boost::make_tuple(0UL, norm_of_rhs);
            }

            size_t     iter = 0;
            value_type res;

            for(; (res = backend::norm(*r) / norm_of_rhs) > tol && iter < maxiter; ++iter)
            {
                P(*r, *s);

                rho2 = rho1;
                rho1 = backend::inner_product(*r, *s);

                if (iter)
                    backend::axpby(1, *s, (rho1 / rho2), *p);
                else
                    backend::copy(*s, *p);

                backend::spmv(1, A, *p, 0, *q);

                value_type alpha = rho1 / backend::inner_product(*q, *p);

                backend::axpby( alpha, *p, 1,  x);
                backend::axpby(-alpha, *q, 1, *r);
            }

            return boost::make_tuple(iter, res);
        }

    private:
        size_t     n;
        size_t     maxiter;
        value_type tol;

        boost::shared_ptr<vector> r;
        boost::shared_ptr<vector> s;
        boost::shared_ptr<vector> p;
        boost::shared_ptr<vector> q;
};

} // namespace solver
} // namespace amgcl


#endif
