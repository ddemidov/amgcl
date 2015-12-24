#ifndef AMGCL_VALUE_TYPE_COMPLEX_HPP
#define AMGCL_VALUE_TYPE_COMPLEX_HPP

/*
The MIT License

Copyright (c) 2012-2015 Christoph Sohrmann

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
 * \file   amgcl/value_type/complex.hpp
 * \author Christoph Sohrmann
 * \brief  Enable std::complex<T> as value type.
 */

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/interface.hpp>

namespace amgcl {
namespace backend {

/// Enable std::complex as a value-type.
template <typename T>
struct is_builtin_vector< std::vector<std::complex<T> > > : boost::true_type {};

} // namespace backend

namespace math {

/// Specialization that extracts the scalar type of a complex type.
template <class T>
struct scalar_of< std::complex<T> > {
    typedef T type;
};

/// Specialization of conjugate transpose for scalar complex arguments.
template <typename T>
struct adjoint_impl< std::complex<T> >
{
    typedef std::complex<T> return_type;

    static std::complex<T> get(std::complex<T> x) {
        return std::conj(x);
    }
};

/// Specialization of zero element for complex type.
template <typename T>
struct is_zero_impl< std::complex<T> >
{
    static bool get(std::complex<T> x) {
        return x == math::make_zero< std::complex<T> >();
    }
};

/// Specialization of zero element for complex type.
template <typename T>
struct make_zero_impl< std::complex<T> >
{
    static std::complex<T> get() {
        return static_cast< std::complex<T> >(0);
    }
};

/// Specialization of identity element for complex type.
template <typename T>
struct make_identity_impl< std::complex<T> >
{
    static std::complex<T> get() {
        return static_cast< std::complex<T> >(1);
    }
};

/// Specialization of constant element for complex type.
template <typename T>
struct make_constant_impl< std::complex<T> >
{
    static std::complex<T> get(T c) {
        return std::complex<T>(c, c);
    }
};

/// Specialization of inversion for complex type.
template <typename T>
struct inverse_impl< std::complex<T> >
{
    static std::complex<T> get(std::complex<T> x) {
        return static_cast<T>(1) / x;
    }
};

}  // namespace math
} // namespace amgcl

namespace std {

template <typename V>
bool operator>(const std::complex<V> &a, const std::complex<V> &b) {
    return std::abs(a) > std::abs(b);
}

template <typename T>
std::complex<T> min(const std::complex<T> &a, const std::complex<T> &b) {
    return std::abs(a) < std::abs(b) ? a : b;
}

template <typename T>
std::complex<T> max(const std::complex<T> &a, const std::complex<T> &b) {
    return std::abs(a) > std::abs(b) ? a : b;
}

} // namespace std

#endif /* ENABLE_COMPLEX_HPP */
