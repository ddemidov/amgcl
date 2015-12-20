#ifndef AMGCL_UTIL_HPP
#define AMGCL_UTIL_HPP

/*
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/util.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Various utilities.
 */

#include <limits>
#include <stdexcept>
#include <boost/io/ios_state.hpp>
#include <boost/property_tree/ptree.hpp>

/* Performance measurement macros
 *
 * If AMGCL_PROFILING macro is defined at compilation, then TIC(name) and
 * TOC(name) macros correspond to prof.tic(name) and prof.toc(name).
 * amgcl::prof should be an instance of amgcl::profiler<> defined in a user
 * code similar to:
 * \code
 * namespace amgcl { profiler<> prof; }
 * \endcode
 * If AMGCL_PROFILING is undefined, then TIC and TOC are noop macros.
 */
#ifdef AMGCL_PROFILING
#  include <amgcl/profiler.hpp>
#  define TIC(name) amgcl::prof.tic(name);
#  define TOC(name) amgcl::prof.toc(name);
namespace amgcl { extern profiler<> prof; }
#else
#  define TIC(name)
#  define TOC(name)
#endif

#define AMGCL_DEBUG_SHOW(x)                                                    \
    std::cout << std::setw(20) << #x << ": "                                   \
              << std::setw(15) << std::setprecision(8) << std::scientific      \
              << (x) << std::endl

#define AMGCL_PARAMS_IMPORT_VALUE(p, name)                                     \
    name( p.get(#name, params().name) )

#define AMGCL_PARAMS_IMPORT_CHILD(p, name)                                     \
    name( p.get_child(#name, amgcl::detail::empty_ptree()) )

#define AMGCL_PARAMS_EXPORT_VALUE(p, path, name)                               \
    p.put(std::string(path) + #name, name)

#define AMGCL_PARAMS_EXPORT_CHILD(p, path, name)                               \
    name.get(p, std::string(path) + #name + ".")

namespace amgcl {

namespace detail {

inline const boost::property_tree::ptree& empty_ptree() {
    static const boost::property_tree::ptree p;
    return p;
}

template <class T>
T eps(size_t n) {
    return 2 * std::numeric_limits<T>::epsilon() * n;
}

} // namespace detail

namespace math {

/// Implementation for conjugate transpose.
/** \note Used in conj_transp() */
template <typename ValueType, class Enable = void>
struct conj_transp_impl {
    typedef typename ValueType::CONJ_TRANSP_NOT_IMPLEMENTED type;
};

/// Implementation for zero check.
/** \note Used in is_zero() */
template <typename ValueType, class Enable = void>
struct is_zero_impl {
    typedef typename ValueType::IS_ZERO_NOT_IMPLEMENTED type;
};

/// Implementation for the zero element.
/** \note Used in make_zero() */
template <typename ValueType, class Enable = void>
struct make_zero_impl {
    typedef typename ValueType::MAKE_ZERO_NOT_IMPLEMENTED type;
};

/// Implementation for the one element.
/** \note Used in make_one() */
template <typename ValueType, class Enable = void>
struct make_one_impl {
    typedef typename ValueType::MAKE_ONE_NOT_IMPLEMENTED type;
};

/// Implementation of inversion operation.
/** \note Used in inverse() */
template <typename ValueType, class Enable = void>
struct inverse_impl {
    typedef typename ValueType::INVERSE_NOT_IMPLEMENTED type;
};

/// Return conjugate transpose of argument.
template <typename ValueType>
ValueType conj_transp(ValueType x) {
	return conj_transp_impl<ValueType>::get(x);
}

/// Return true if argument is considered zero.
template <typename ValueType>
bool is_zero(ValueType x) {
	return is_zero_impl<ValueType>::get(x);
}

/// Create zero element of type ValueType.
template <typename ValueType>
ValueType make_zero() {
	return make_zero_impl<ValueType>::get();
}

/// Create one element of type ValueType.
template <typename ValueType>
ValueType make_one() {
	return make_one_impl<ValueType>::get();
}

/// Return inverse of the argument.
template <typename ValueType>
ValueType inverse(ValueType x) {
	return inverse_impl<ValueType>::get(x);
}

template <typename ValueType>
struct conj_transp_impl<ValueType,
typename boost::enable_if<boost::is_arithmetic<ValueType> >::type>
{
    /// Conjuate transpose is noop for arithmetic types.
	static ValueType get(ValueType x) {
		return x;
	}
};

template <typename ValueType>
struct is_zero_impl<ValueType,
typename boost::enable_if<boost::is_arithmetic<ValueType> >::type>
{
	static bool get(ValueType x) {
		return x == make_zero<ValueType>();
	}
};

template <typename ValueType>
struct make_zero_impl<ValueType,
typename boost::enable_if<boost::is_arithmetic<ValueType> >::type>
{
	static ValueType get() {
		return static_cast<ValueType>(0);
	}
};

template <typename ValueType>
struct make_one_impl<ValueType,
typename boost::enable_if<boost::is_arithmetic<ValueType> >::type>
{
	static ValueType get() {
		return static_cast<ValueType>(1);
	}
};

template <typename ValueType>
struct inverse_impl<ValueType,
typename boost::enable_if<boost::is_arithmetic<ValueType> >::type>
{
	static ValueType get(ValueType x) {
		return make_one<ValueType>() / x;
	}
};

}  // namespace math

/// Throws \p message if \p condition is not true.
template <class Condition, class Message>
void precondition(const Condition &condition, const Message &message) {
    if ( !static_cast<bool>(condition) )
        throw std::runtime_error(message);
}

} // namespace amgcl

namespace std {

// Read pointers from input streams.
// This allows to exchange pointers through boost::property_tree::ptree.
template <class T>
inline istream& operator>>(istream &is, T* &ptr) {
    boost::io::ios_all_saver stream_state(is);

    size_t val;
    is >> std::hex >> val;

    ptr = reinterpret_cast<T*>(val);

    return is;
}

} // namespace std


#endif
