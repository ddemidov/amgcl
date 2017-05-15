#ifndef AMGCL_RELAXATION_DETAIL_ILU_SOLVE_HPP
#define AMGCL_RELAXATION_DETAIL_ILU_SOLVE_HPP

#include <amgcl/backend/interface.hpp>

namespace amgcl {
namespace relaxation {
namespace detail {

template <class Backend> struct ilu_solve {
    typedef typename Backend::value_type value_type;
    typedef typename Backend::matrix matrix;
    typedef typename Backend::vector vector;
    typedef typename Backend::matrix_diagonal matrix_diagonal;
    typedef typename backend::builtin<value_type>::matrix build_matrix;
    typedef typename math::scalar_of<value_type>::type scalar_type;

    unsigned jacobi_iters;

    boost::shared_ptr<matrix> L;
    boost::shared_ptr<matrix> U;
    boost::shared_ptr<matrix_diagonal> D;
    boost::shared_ptr<vector> t1, t2;

    template <class Params>
    ilu_solve(
            boost::shared_ptr<build_matrix> L,
            boost::shared_ptr<build_matrix> U,
            boost::shared_ptr<backend::numa_vector<value_type> > D,
            const Params &prm,
            const typename Backend::params &bprm
            ) :
        jacobi_iters(prm.jacobi_iters),
        L(Backend::copy_matrix(L, bprm)),
        U(Backend::copy_matrix(U, bprm)),
        D(Backend::copy_vector(D, bprm)),
        t1(Backend::create_vector(backend::rows(*L), bprm)),
        t2(Backend::create_vector(backend::rows(*L), bprm))
    {}

    template <class Vector>
    void solve(Vector &x) {
        vector *y0 = t1.get();
        vector *y1 = t2.get();

        backend::copy(x, *y0);
        for(unsigned i = 0; i < jacobi_iters; ++i) {
            backend::residual(x, *L, *y0, *y1);
            std::swap(y0, y1);
        }

        backend::copy(*y0, x);
        for(unsigned i = 0; i < jacobi_iters; ++i) {
            backend::residual(*y0, *U, x, *y1);
            backend::vmul(math::identity<scalar_type>(), *D, *y1, math::zero<scalar_type>(), x);
        }
    }
};

template <class value_type>
struct ilu_solve< backend::builtin<value_type> > {
    typedef backend::builtin<value_type> Backend;
    typedef typename Backend::matrix matrix;
    typedef typename Backend::vector vector;
    typedef typename Backend::matrix_diagonal matrix_diagonal;
    typedef typename backend::builtin<value_type>::matrix build_matrix;
    typedef typename math::scalar_of<value_type>::type scalar_type;

    boost::shared_ptr<matrix> L;
    boost::shared_ptr<matrix> U;
    boost::shared_ptr<matrix_diagonal> D;

    template <class Params>
    ilu_solve(
            boost::shared_ptr<build_matrix> L,
            boost::shared_ptr<build_matrix> U,
            boost::shared_ptr<backend::numa_vector<value_type> > D,
            const Params&, const typename Backend::params&
            ) : L(L), U(U), D(D)
    {}

    template <class Vector>
    void solve(Vector &x) {
        const size_t n = backend::rows(*L);

        for(size_t i = 0; i < n; i++) {
            for(ptrdiff_t j = L->ptr[i], e = L->ptr[i+1]; j < e; ++j)
                x[i] -= L->val[j] * x[L->col[j]];
        }

        for(size_t i = n; i-- > 0;) {
            for(ptrdiff_t j = U->ptr[i], e = U->ptr[i+1]; j < e; ++j)
                x[i] -= U->val[j] * x[U->col[j]];
            x[i] = (*D)[i] * x[i];
        }
    }
};

} // namespace detail
} // namespace relaxation
} // namespace amgcl

#endif
