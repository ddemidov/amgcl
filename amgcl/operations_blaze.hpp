#ifndef AMGCL_OPERATIONS_BLAZE_HPP
#define AMGCL_OPERATIONS_BLAZE_HPP

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
 * \file   operations_blaze.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Adaptors for Blaze types.
 */

#include <type_traits>
#include <amgcl/common.hpp>
#include <blaze/Math.h>

namespace amgcl {

/// Returns value type of vex::vector.
/** Necessary for blaze types to work with amgcl::solve() functions. */
template <typename T>
struct value_type<T,
    typename std::enable_if<std::is_arithmetic<typename T::MatrixType>::value>::type
    >
{
    typedef typename T::MatrixType type;
};

template <typename T>
struct value_type< blaze::DynamicVector<T> > {
    typedef T type;
};

/// Returns inner product of two vex::vectors.
/** Necessary for blaze types to work with amgcl::solve() functions. */
template <typename T>
T inner_prod(const blaze::DynamicVector<T> &x, const blaze::DynamicVector<T> &y) {
    return (x, y);
}

/// Returns norm of vex::vector.
/** Necessary for blaze types to work with amgcl::solve() functions. */
template <typename T>
T norm(const blaze::DynamicVector<T> &x) {
    return sqrt( (x, x) );
}

/// Clears (sets elements to zero) vex::vector.
/** Necessary for blaze types to work with amgcl::solve() functions. */
template <typename T>
void clear(blaze::DynamicVector<T> &x) {
    x = 0;
}

} // namespace amgcl

#endif
