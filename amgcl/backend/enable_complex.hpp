#ifndef AMGCL_BACKEND_ENABLE_COMPLEX_HPP
#define AMGCL_BACKEND_ENABLE_COMPLEX_HPP

#include <complex>
#include <boost/type_traits.hpp>

namespace std {

template <typename T, typename U>
inline typename boost::enable_if<
    typename boost::is_convertible<T, U>::type,
    std::complex<U>
    >::type
operator*(T a, std::complex<U> b) {
    return static_cast<U>(a) * b;
}

template <typename T, typename U>
inline typename boost::enable_if<
    typename boost::is_convertible<T, U>::type,
    std::complex<U>
    >::type
operator/(T a, std::complex<U> b) {
    return static_cast<U>(a) / b;
}

}

#endif
