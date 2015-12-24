#ifndef AMGCL_VALUE_TYPE_INTERFACE_HPP
#define AMGCL_VALUE_TYPE_INTERFACE_HPP

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
 * \file   amgcl/value_type/interface.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Support for various value types.
 */

namespace amgcl {
namespace math {

/// Implementation for conjugate transpose.
/** \note Used in adjoint() */
template <typename ValueType, class Enable = void>
struct adjoint_impl {
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
typename adjoint_impl<ValueType>::return_type
adjoint(ValueType x) {
    return adjoint_impl<ValueType>::get(x);
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
struct adjoint_impl<ValueType,
typename boost::enable_if<boost::is_arithmetic<ValueType> >::type>
{
    typedef ValueType return_type;

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


} // namespace math
} // namespace amgcl

#endif
