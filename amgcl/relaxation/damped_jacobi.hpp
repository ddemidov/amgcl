#ifndef AMGCL_RELAXATION_DAMPED_JACOBI_HPP
#define AMGCL_RELAXATION_DAMPED_JACOBI_HPP

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
 * \file   amgcl/relaxation/damped_jacobi.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Damped Jacobi relaxation scheme.
 */

#include <amgcl/backend/interface.hpp>
#include <amgcl/relaxation/interface.hpp>

namespace amgcl {
namespace relaxation {

template <class Backend>
struct impl<damped_jacobi, Backend> {
    struct params {
        typename Backend::value_type damping;

        params(typename Backend::value_type damping = 0.72)
            : damping(damping) {}
    };

    boost::shared_ptr<typename Backend::vector> dia;

    template <class Matrix>
    impl( const Matrix &A, const params &, const typename Backend::params &backend_prm)
        : dia( Backend::copy_vector( diagonal(A, true), backend_prm ) )
    {}

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        backend::residual(rhs, A, x, tmp);
        backend::vmul(prm.damping, *dia, tmp, 1, x);
    }
    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply(A, rhs, x, tmp, prm);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply(A, rhs, x, tmp, prm);
    }
};

} // namespace relaxation
} // namespace amgcl

#endif
