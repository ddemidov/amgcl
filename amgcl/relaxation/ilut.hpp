#ifndef AMGCL_RELAXATION_ILUT_HPP
#define AMGCL_RELAXATION_ILUT_HPP

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
 * \file   amgcl/relaxation/ilut.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Incomplete LU with thresholding relaxation scheme.
 */

#include <boost/foreach.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace relaxation {

namespace detail {

template <typename T>
class sparse_vector {
    public:
        sparse_vector(size_t n) : idx(n, -1) {
            val.reserve(16);
        }

        T operator[](ptrdiff_t i) const {
            if (idx[i] >= 0) return val[idx[i]].first;
            return T();
        }

        T& operator[](ptrdiff_t i) {
            if (idx[i] == -1) {
                idx[i] = val.size();
                val.push_back(std::make_pair(T(), i));
            }
            return val[idx[i]].first;
        }

        void clear() {
            BOOST_FOREACH(const nonzero &e, val) idx[e.second] = -1;
            val.clear();
        }
    private:
        typedef std::pair<T, ptrdiff_t> nonzero;

        std::vector<nonzero>   val;
        std::vector<ptrdiff_t> idx;
};

} // namespace detail

/// ILUT(p, tau) smoother.
/**
 * \note ILUT is a serial algorithm and is only applicable to backends that
 * support matrix row iteration (e.g. amgcl::backend::builtin or
 * amgcl::backend::eigen).
 *
 * \param Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 */
template <class Backend>
struct ilut {
    typedef typename Backend::value_type value_type;
    typedef typename Backend::vector     vector;

    /// Relaxation parameters.
    struct params {
        /// Damping factor.
        float damping;

        /// Maximum fill-in.
        int p;

        /// Minimum magnitude of non-zero elements relative to the current row norm.
        float tau;

        params(int p = 2, float tau = 1e-2f, float damping = 1)
            : p(p), tau(tau), damping(damping) {}

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_VALUE(p, p)
            , AMGCL_PARAMS_IMPORT_VALUE(p, tau)
            , AMGCL_PARAMS_IMPORT_VALUE(p, damping)
        {}

        void get(boost::property_tree::ptree &p, const std::string &path) const {
            AMGCL_PARAMS_EXPORT_VALUE(p, path, p);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, tau);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, damping);
        }
    };

    /// \copydoc amgcl::relaxation::damped_jacobi::damped_jacobi
    template <class Matrix>
    ilut( const Matrix &A, const params &prm, const typename Backend::params&)
          : dia( backend::rows(A) )
    {
        LU.reserve(2 * prm.p * backend::rows(A) + backend::nonzeros(A));
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

    private:
        std::vector<value_type> LU;
        std::vector<ptrdiff_t>  dia;

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
                for(ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
                    buf -= A.val[j] * x[A.col[j]];
                tmp[i] = buf;
            }

            for(size_t i = 0; i < n; i++) {
                for(ptrdiff_t j = A.ptr[i], e = dia[i]; j < e; ++j)
                    x[i] -= LU[j] * x[A.col[j]];
            }

            for(size_t i = n; i-- > 0;) {
                for(ptrdiff_t j = dia[i] + 1, e = A.ptr[i + 1]; j < e; ++j)
                    x[i] -= LU[j] * x[A.col[j]];
                x[i] *= LU[dia[i]];
            }

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
                x[i] += prm.damping * tmp[i];
        }
};

} // namespace relaxation
} // namespace amgcl

#endif
