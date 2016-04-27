#ifndef AMGCL_RELAXATION_PASTIX_ILU_HPP
#define AMGCL_RELAXATION_PASTIX_ILU_HPP

#include <boost/typeof/typeof.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/relaxation/detail/ilu_solve.hpp>

extern "C" {
#include <pastix.h>
}

namespace amgcl {
namespace relaxation {

template <class Backend>
struct pastix_ilu {
    typedef typename Backend::value_type      value_type;
    typedef typename Backend::vector          vector;
    typedef typename Backend::matrix          matrix;
    typedef typename Backend::matrix_diagonal matrix_diagonal;

    typedef typename math::scalar_of<value_type>::type scalar_type;

    /// Relaxation parameters.
    struct params {
        int lfil;

        /// Damping factor.
        scalar_type damping;

        params() : lfil(1), damping(1) {}

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_VALUE(p, lfil)
            , AMGCL_PARAMS_IMPORT_VALUE(p, damping)
        {}

        void get(boost::property_tree::ptree &p, const std::string &path) const {
            AMGCL_PARAMS_EXPORT_VALUE(p, path, lfil);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, damping);
        }
    };

    /// \copydoc amgcl::relaxation::damped_jacobi::damped_jacobi
    template <class Matrix>
    pastix_ilu( const Matrix &A, const params &prm, const typename Backend::params &bprm)
        : n(backend::rows(A)), perm(n), invp(n)
    {
        BOOST_AUTO(Aptr, A.ptr_data());
        BOOST_AUTO(Acol, A.col_data());
        BOOST_AUTO(Aval, A.val_data());

        ptr.assign(Aptr, Aptr + n + 1);
        col.assign(Acol, Acol + Aptr[n]);
        val.assign(Aval, Aval + Aptr[n]);

        BOOST_FOREACH(pastix_int_t &p, ptr) ++p;
        BOOST_FOREACH(pastix_int_t &c, col) ++c;

        // Initialize parameters with default values:
        iparm[IPARM_MODIFY_PARAMETER] = API_NO;
        call_pastix(API_TASK_INIT, API_TASK_INIT);

        // Factorize the matrix.
        iparm[IPARM_VERBOSE        ] = API_VERBOSE_NOT;
        iparm[IPARM_RHS_MAKING     ] = API_RHS_B;
        iparm[IPARM_SYM            ] = API_SYM_NO;
        iparm[IPARM_FACTORIZATION  ] = API_FACT_LU;
        iparm[IPARM_TRANSPOSE_SOLVE] = API_YES;
        iparm[IPARM_INCOMPLETE     ] = API_YES;
        iparm[IPARM_LEVEL_OF_FILL  ] = prm.lfil;

#ifdef _OPENMP
        iparm[IPARM_THREAD_NBR]      = omp_get_max_threads();
#endif
        call_pastix(API_TASK_ORDERING, API_TASK_NUMFACT);
    }

    /// \copydoc amgcl::relaxation::damped_jacobi::apply_pre
    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
    }

    /// \copydoc amgcl::relaxation::damped_jacobi::apply_post
    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
    }

    /// \copydoc amgcl::relaxation::damped_jacobi::apply_post
    template <class Matrix, class VectorRHS, class VectorX>
    void apply(const Matrix &A, const VectorRHS &rhs, VectorX &x, const params &prm) const
    {
        backend::copy(rhs, x);
        call_pastix(API_TASK_SOLVE, API_TASK_SOLVE, &x[0]);
    }

    private:
        size_t n;
        std::vector<pastix_int_t> ptr;
        std::vector<pastix_int_t> col;
        std::vector<value_type>   val;

        // Pastix internal data.
        mutable pastix_data_t *pastix_data;

        // Pastix parameters
        mutable pastix_int_t   iparm[IPARM_SIZE];
        mutable double         dparm[DPARM_SIZE];

        std::vector<pastix_int_t> perm, invp;

        void call_pastix(int beg, int end, value_type *x = NULL) const {
            iparm[IPARM_START_TASK] = beg;
            iparm[IPARM_END_TASK  ] = end;

            call_pastix(x);
        }

        void call_pastix(double *x) const {
            d_pastix(&pastix_data, MPI_COMM_WORLD, n,
                    const_cast<pastix_int_t*>(&ptr[0]),
                    const_cast<pastix_int_t*>(&col[0]),
                    const_cast<double*      >(&val[0]),
                    const_cast<pastix_int_t*>(&perm[0]),
                    const_cast<pastix_int_t*>(&invp[0]),
                    x, 1, iparm, dparm
                   );
        }

        void call_pastix(float *x) const {
            s_pastix(&pastix_data, MPI_COMM_WORLD, n,
                    const_cast<pastix_int_t*>(&ptr[0]),
                    const_cast<pastix_int_t*>(&col[0]),
                    const_cast<float*       >(&val[0]),
                    const_cast<pastix_int_t*>(&perm[0]),
                    const_cast<pastix_int_t*>(&invp[0]),
                    x, 1, iparm, dparm
                   );
        }
};

} // namespace relaxation

namespace backend {

template <class Backend>
struct relaxation_is_supported<
    Backend,
    relaxation::pastix_ilu
    > : boost::false_type
{};

template <>
struct relaxation_is_supported<
    backend::builtin<double>,
    relaxation::pastix_ilu
    > : boost::true_type
{};

} // namespace backend
} // namespace amgcl

#endif
