#ifndef ENABLE_COMPLEX_HPP
#define ENABLE_COMPLEX_HPP

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
 * \file   amgcl/backend/enable_complex.hpp
 * \author Christoph Sohrmann
 * \brief  Enable std::complex<T> as value type.
 */

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace backend {

/// Enable std::complex as a value-type.
template <typename T>
struct is_builtin_vector< std::vector<std::complex<T> > > : boost::true_type {};

/// Specialization that extracts the scalar type of a complex type.
template <class T>
struct scalar_of< std::complex<T> > {
    typedef T type;
};

} // namespace backend

namespace math {

/// Specialization of conjugate transpose for scalar complex arguments.
template <typename ValueType>
struct conj_transp_impl<
    ValueType,
    typename boost::enable_if<boost::is_complex<ValueType> >::type
    >
{
    static ValueType get(ValueType x) {
        return std::conj(x);
    }
};

/// Specialization of zero element for complex type.
template <typename ValueType>
struct is_zero_impl<ValueType,
typename boost::enable_if<boost::is_complex<ValueType> >::type>
{
    static bool get(ValueType x) {
        return x == math::make_zero<ValueType>();
    }
};

/// Specialization of zero element for complex type.
template <typename ValueType>
struct make_zero_impl<ValueType,
typename boost::enable_if<boost::is_complex<ValueType> >::type>
{
    static ValueType get() {
        return static_cast<ValueType>(0);
    }
};

/// Specialization of one element for complex type.
template <typename ValueType>
struct make_one_impl<ValueType,
typename boost::enable_if<boost::is_complex<ValueType> >::type>
{
    static ValueType get() {
        return static_cast<ValueType>(1);
    }
};

/// Specialization of inversion for complex type.
template <typename ValueType>
struct inverse_impl<ValueType,
typename boost::enable_if<boost::is_complex<ValueType> >::type>
{
    static ValueType get(ValueType x) {
        return math::make_one<ValueType>() / x;
    }
};

}  // namespace math

template <typename V>
bool operator>(const std::complex<V> &a, const std::complex<V> &b) {
    return std::abs(a) > std::abs(b);
}

} // namespace amgcl

namespace std {

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
