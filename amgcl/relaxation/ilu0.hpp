#ifndef AMGCL_RELAXATION_ILU0_HPP
#define AMGCL_RELAXATION_ILU0_HPP

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
 * \file   amgcl/relaxation/ilu0.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Incomplete LU with zero fill-in relaxation scheme.
 */

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace relaxation {

/// ILU(0) smoother.
/**
 * \note ILU(0) is a serial algorithm and is only applicable to backends that
 * support matrix row iteration (e.g. amgcl::backend::builtin or
 * amgcl::backend::eigen).
 *
 * \param Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 */
template <class Backend>
struct ilu0 {
    typedef typename Backend::value_type value_type;
    typedef typename Backend::vector     vector;
    typedef typename Backend::matrix     matrix;

    /// Relaxation parameters.
    struct params {
        /// Damping factor.
        float damping;

        /// Number of Jacobi iterations.
        /** \note Used for approximate solution of triangular systems on parallel backends */
        unsigned jacobi_iters;

        params(float damping = 1, unsigned jacobi_iters = 2)
            : damping(damping), jacobi_iters(jacobi_iters) {}

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_VALUE(p, damping)
            , AMGCL_PARAMS_IMPORT_VALUE(p, jacobi_iters)
        {}

        void get(boost::property_tree::ptree &p, const std::string &path) const {
            AMGCL_PARAMS_EXPORT_VALUE(p, path, damping);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, jacobi_iters);
        }
    };

    /// \copydoc amgcl::relaxation::damped_jacobi::damped_jacobi
    template <class Matrix>
    ilu0( const Matrix &A, const params &, const typename Backend::params &bprm)
    {
        typedef typename backend::builtin<value_type>::matrix build_matrix;
        const size_t n = backend::rows(A);

        boost::shared_ptr<build_matrix> L = boost::make_shared<build_matrix>();
        boost::shared_ptr<build_matrix> U = boost::make_shared<build_matrix>();

        L->nrows = L->ncols = n;
        U->nrows = U->ncols = n;

        L->ptr.reserve(n+1); L->ptr.push_back(0);
        U->ptr.reserve(n+1); U->ptr.push_back(0);

        std::vector<value_type> D;
        D.reserve(n);

        size_t Lnz = 0, Unz = 0;

        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            ptrdiff_t row_beg = A.ptr[i];
            ptrdiff_t row_end = A.ptr[i + 1];

            for(ptrdiff_t j = row_beg; j < row_end; ++j) {
                ptrdiff_t c = A.col[j];
                if (c < i)
                    ++Lnz;
                else if (c > i)
                    ++Unz;
            }
        }

        L->col.reserve(Lnz);
        L->val.reserve(Lnz);

        U->col.reserve(Unz);
        U->val.reserve(Unz);

        std::vector<value_type*> work(n, NULL);

        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            ptrdiff_t row_beg = A.ptr[i];
            ptrdiff_t row_end = A.ptr[i + 1];

            for(ptrdiff_t j = row_beg; j < row_end; ++j) {
                ptrdiff_t  c = A.col[j];
                value_type v = A.val[j];

                if (c < i) {
                    L->col.push_back(c);
                    L->val.push_back(v);
                    work[c] = &L->val.back();
                } else if (c == i) {
                    D.push_back(v);
                    work[c] = &D.back();
                } else {
                    U->col.push_back(c);
                    U->val.push_back(v);
                    work[c] = &U->val.back();
                }
            }

            L->ptr.push_back(L->val.size());
            U->ptr.push_back(U->val.size());

            for(ptrdiff_t j = row_beg; j < row_end; ++j) {
                ptrdiff_t c = A.col[j];

                // Exit if diagonal is reached
                if (c >= i) {
                    precondition(c == i, "No diagonal value in system matrix");
                    precondition(D[i] != 0, "Zero pivot in ILU");

                    D[i] = 1 / D[i];
                    break;
                }

                // Compute the multiplier for jrow
                value_type tl = (*work[c]) * D[c];
                *work[c] = tl;

                // Perform linear combination
                for(ptrdiff_t k = U->ptr[c]; k < U->ptr[c+1]; ++k) {
                    value_type *w = work[U->col[k]];
                    if (w) *w -= tl * U->val[k];
                }
            }

            // Refresh work
            for(ptrdiff_t j = row_beg; j < row_end; ++j)
                work[A.col[j]] = NULL;
        }

        this->D = Backend::copy_vector(D, bprm);
        this->L = Backend::copy_matrix(L, bprm);
        this->U = Backend::copy_matrix(U, bprm);

        if (!boost::is_same<Backend, backend::builtin<value_type> >::value) {
            y0 = Backend::create_vector(n, bprm);
            y1 = Backend::create_vector(n, bprm);
        }
    }

    /// \copydoc amgcl::relaxation::damped_jacobi::apply_pre
    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply<Backend>(A, rhs, x, tmp, prm);
    }

    /// \copydoc amgcl::relaxation::damped_jacobi::apply_post
    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply<Backend>(A, rhs, x, tmp, prm);
    }

    private:
        boost::shared_ptr<matrix> L, U;
        boost::shared_ptr<vector> D, y0, y1;

        template <class B, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
        typename boost::enable_if<
            typename boost::is_same<
                B, backend::builtin<typename B::value_type>
            >::type,
            void
        >::type
        apply(
                const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
                const params &prm
                ) const
        {
            const size_t n = backend::rows(A);

            backend::residual(rhs, A, x, tmp);

            for(size_t i = 0; i < n; i++) {
                for(ptrdiff_t j = L->ptr[i], e = L->ptr[i+1]; j < e; ++j)
                    tmp[i] -= L->val[j] * tmp[L->col[j]];
            }

            for(size_t i = n; i-- > 0;) {
                for(ptrdiff_t j = U->ptr[i], e = U->ptr[i+1]; j < e; ++j)
                    tmp[i] -= U->val[j] * tmp[U->col[j]];
                tmp[i] *= (*D)[i];
            }

            backend::axpby(prm.damping, tmp, 1, x);
        }

        template <class B, class Matrix, class VectorRHS, class VectorX, class VectorTMP>
        typename boost::disable_if<
            typename boost::is_same<
                B, backend::builtin<typename B::value_type>
            >::type,
            void
        >::type
        apply(
                const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
                const params &prm
                ) const
        {
            backend::residual(rhs, A, x, tmp);
            backend::copy(tmp, *y0);

            // Solve Ly = b through Jacobi iterations
            for (unsigned i = 0; i < prm.jacobi_iters; ++i) {
                backend::residual(tmp, *L, *y0, *y1);
                backend::copy(*y1, *y0);
            }

            // Solve Ux = y through Jacobi iterations
            backend::copy(*y1, tmp);
            for(unsigned i = 0; i < prm.jacobi_iters; ++i) {
                backend::residual(tmp, *U, *y0, *y1);
                backend::vmul(1, *D, *y1, 0, *y0);
            }

            backend::axpby(prm.damping, *y0, 1, x);
        }

};

} // namespace relaxation
} // namespace amgcl



#endif
