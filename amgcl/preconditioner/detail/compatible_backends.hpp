#ifndef AMGCL_PRECONDITIONER_DETAIL_COMPATIBLE_BACKENDS_HPP
#define AMGCL_PRECONDITIONER_DETAIL_COMPATIBLE_BACKENDS_HPP

#include <boost/type_traits.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace preconditioner {
namespace detail {

// Same backends are always compatible
template <class B1, class B2>
struct compatible_backends
    : boost::is_same<B1, B2>::type {};

// Builtin backend allows mixing backends of different value types,
// so that scalar and non-scalar backends may coexist.
template <class V1, class V2>
struct compatible_backends< backend::builtin<V1>, backend::builtin<V2> >
    : boost::true_type {};

// Backend for schur complement preconditioner is selected as the one with
// lower dimensionality of its value_type.

template <class B1, class B2, class Enable = void>
struct common_backend;

template <class B>
struct common_backend<B, B> {
    typedef B type;
};

template <class V1, class V2>
struct common_backend< backend::builtin<V1>, backend::builtin<V2>,
    typename boost::disable_if<typename boost::is_same<V1, V2>::type>::type >
{
    typedef
        typename boost::conditional<
            (math::static_rows<V1>::value <= math::static_rows<V2>::value),
            backend::builtin<V1>, backend::builtin<V2>
            >::type
        type;
};

} // namespace detail
} // namespace preconditioner
} // namespace amgcl



#endif
