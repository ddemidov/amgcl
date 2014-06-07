#ifndef AMGCL_RELAXATION_ILU0_HPP
#define AMGCL_RELAXATION_ILU0_HPP

/**
 * \file   amgcl/relaxation/ilu0.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Incomplete LU relaxation scheme.
 */

#include <boost/shared_ptr.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace relaxation {

template <class Backend>
struct ilu0 {
    typedef typename Backend::value_type value_type;
    typedef typename Backend::vector     vector;

    struct params {
        float damping;
        params(float damping = 0.72) : damping(damping) {}
    };

    std::vector<value_type> luval;
    std::vector<long>       dia;

    template <class Matrix>
    ilu0( const Matrix &A, const params &, const typename Backend::params&)
        : luval( A.val ),
          dia  ( backend::rows(A) )
    {
        const size_t n = backend::rows(A);

        std::vector<long> work(n, -1);

        for(size_t i = 0; i < n; ++i) {
            long row_beg = A.ptr[i];
            long row_end = A.ptr[i + 1];

            for(long j = row_beg; j < row_end; ++j)
                work[ A.col[j] ] = j;

            for(long j = row_beg; j < row_end; ++j) {
                long c = A.col[j];

                // Exit if diagonal is reached
                if (static_cast<size_t>(c) >= i) {
                    precondition(static_cast<size_t>(c) == i,
                            "No diagonal value in system matrix");
                    precondition(fabs(luval[j]) > 1e-32,
                            "Zero pivot in ILU");

                    dia[i]   = j;
                    luval[j] = 1 / luval[j];
                    break;
                }

                // Compute the multiplier for jrow
                value_type tl = luval[j] * luval[dia[c]];
                luval[j] = tl;

                // Perform linear combination
                for(long k = dia[c] + 1; k < A.ptr[c + 1]; ++k) {
                    long w = work[A.col[k]];
                    if (w >= 0) luval[w] -= tl * luval[k];
                }
            }

            // Refresh work
            for(long j = row_beg; j < row_end; ++j)
                work[A.col[j]] = -1;
        }
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        const size_t n = backend::rows(A);

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            value_type buf = rhs[i];
            for(long j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
                buf -= A.val[j] * x[A.col[j]];
            tmp[i] = buf;
        }

        for(size_t i = 0; i < n; i++) {
            for(long j = A.ptr[i], e = dia[i]; j < e; ++j)
                tmp[i] -= luval[j] * tmp[A.col[j]];
        }

        for(size_t i = n; i-- > 0;) {
            for(long j = dia[i] + 1, e = A.ptr[i + 1]; j < e; ++j)
                tmp[i] -= luval[j] * tmp[A.col[j]];
            tmp[i] *= luval[dia[i]];
        }

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
            x[i] += prm.damping * tmp[i];
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply(A, rhs, x, tmp, prm);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply(A, rhs, x, tmp, prm);
    }
};

} // namespace relaxation
} // namespace amgcl



#endif
