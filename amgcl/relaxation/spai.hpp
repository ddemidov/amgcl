#ifndef AMGCL_RELAXATION_SPAI_HPP
#define AMGCL_RELAXATION_SPAI_HPP

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
 * \file   amgcl/relaxation/spai.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Sparse approximate inverse relaxation scheme.
 */

#include <amgcl/backend/interface.hpp>
#include <amgcl/relaxation/interface.hpp>

namespace amgcl {
namespace relaxation {

template <class Backend>
struct impl<spai0, Backend> {
    struct params {};

    boost::shared_ptr<typename Backend::vector> M;

    template <class Matrix>
    impl( const Matrix &A, const params &, const typename Backend::params &backend_prm)
    {
        typedef typename backend::value_type<Matrix>::type   V;
        typedef typename backend::row_iterator<Matrix>::type row_iterator;

        const size_t n = rows(A);

        std::vector<V> m(n);

#pragma omp parallel for
        for(size_t i = 0; i < n; ++i) {
            V num = 0;
            V den = 0;

            for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                V v = a.value();
                den += v * v;
                if (static_cast<size_t>(a.col()) == i) num += v;
            }

            m[i] = num / den;
        }

        M = Backend::copy_vector(m, backend_prm);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp
            ) const
    {
        backend::residual(rhs, A, x, tmp);
        backend::vmul(1, *M, tmp, 1, x);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params&
            ) const
    {
        apply(A, rhs, x, tmp);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params&
            ) const
    {
        apply(A, rhs, x, tmp);
    }
};

} // namespace relaxation
} // namespace amgcl

#endif
