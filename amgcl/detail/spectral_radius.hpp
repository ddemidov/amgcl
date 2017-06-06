#ifndef AMGCL_DETAIL_SPECTRAL_RADIUS_HPP
#define AMGCL_DETAIL_SPECTRAL_RADIUS_HPP

/*
The MIT License

Copyright (c) 2012-2017 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/detail/spectral_radius.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Estimate spectral radius of a matrix.
 */

#include <algorithm>
#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>

namespace amgcl {
namespace detail {

template <class Matrix>
inline
typename math::scalar_of<typename backend::value_type<Matrix>::type>::type
spectral_radius(const Matrix &A, bool scale_by_dia = false)
{
    typedef typename backend::value_type<Matrix>::type   value_type;
    typedef typename math::scalar_of<value_type>::type   scalar_type;
    typedef typename backend::row_iterator<Matrix>::type row_iterator;

    const ptrdiff_t n = backend::rows(A);

    scalar_type emax = 0;

#pragma omp parallel
    {
        scalar_type my_emax = 0;
#pragma omp for nowait
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            scalar_type hi  = 0;
            scalar_type dia = 1;

            for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                scalar_type v = math::norm(a.value());
                if (scale_by_dia && a.col() == i) dia = v;
                hi += v;
            }

            if (scale_by_dia) hi = math::inverse(dia) * hi;

            my_emax = std::max(my_emax, hi);
        }

#pragma omp critical
        emax = std::max(emax, my_emax);
    }

    return emax;
}

} // namespace detail
} // namespace amgcl

#endif
