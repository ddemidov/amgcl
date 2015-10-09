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

#include <vector>
#include <queue>
#include <cmath>

#include <boost/foreach.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace relaxation {

namespace detail {

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
        /// Maximum fill-in.
        int p;

        /// Minimum magnitude of non-zero elements relative to the current row norm.
        float tau;

        /// Damping factor.
        float damping;

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
    ilut( const Matrix &A, const params &prm, const typename Backend::params&) {
        const size_t n = backend::rows(A);
        typedef typename backend::row_iterator<Matrix>::type row_iterator;

        LU.ncols = n;
        LU.nrows = n;

        LU.col.reserve(backend::nonzeros(A) + 2 * prm.p * n);
        LU.val.reserve(backend::nonzeros(A) + 2 * prm.p * n);
        LU.ptr.reserve(n + 1);
        LU.ptr.push_back(0);

        uptr.reserve(n);

        sparse_vector w(n);

        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            w.dia = i;

            int lenL = 0;
            int lenU = 0;

            value_type tol = 0;

            for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                w[a.col()] = a.value();
                tol += fabs(a.value());

                if (a.col() <  i) ++lenL;
                if (a.col() >= i) ++lenU;
            }
            tol = prm.tau / (lenL + lenU);

            while(!w.q.empty()) {
                ptrdiff_t k = w.next_nonzero();
                value_type wk = (w[k] *= LU.val[uptr[k]]);

                if (fabs(wk) > tol) {
                    for(ptrdiff_t j = uptr[k] + 1; j < LU.ptr[k+1]; ++j) {
                        w[LU.col[j]] -= wk * LU.val[j];
                    }
                }
            }

            w.move_to(lenL + prm.p, lenU + prm.p, tol, LU, uptr);
        }
    }

    /// \copydoc amgcl::relaxation::damped_jacobi::apply_pre
    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply(A, rhs, x, tmp, prm);
    }

    /// \copydoc amgcl::relaxation::damped_jacobi::apply_post
    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(
            const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
            const params &prm
            ) const
    {
        apply(A, rhs, x, tmp, prm);
    }

    private:
        typedef typename backend::builtin<value_type>::matrix build_matrix;

        build_matrix LU;
        std::vector<ptrdiff_t>  uptr;

        struct sparse_vector {
            struct nonzero {
                ptrdiff_t  col;
                value_type val;

                nonzero() : col(-1) {}

                nonzero(ptrdiff_t col, value_type val = value_type())
                    : col(col), val(val) {}
            };

            struct comp_indices {
                const std::vector<nonzero> &nz;

                comp_indices(const std::vector<nonzero> &nz) : nz(nz) {}

                bool operator()(int a, int b) const {
                    return nz[a].col > nz[b].col;
                }
            };

            typedef
                std::priority_queue<int, std::vector<int>, comp_indices>
                priority_queue;

            std::vector<nonzero>   nz;
            std::vector<ptrdiff_t> idx;
            priority_queue q;

            ptrdiff_t dia;

            sparse_vector(size_t n) : idx(n, -1), q(comp_indices(nz)), dia(0) {
                nz.reserve(16);
            }

            value_type operator[](ptrdiff_t i) const {
                if (idx[i] >= 0) return nz[idx[i]].val;
                return value_type();
            }

            value_type& operator[](ptrdiff_t i) {
                if (idx[i] == -1) {
                    int p = nz.size();
                    idx[i] = p;
                    nz.push_back(nonzero(i));
                    if (i < dia) q.push(p);
                }
                return nz[idx[i]].val;
            }

            typename std::vector<nonzero>::iterator begin() {
                return nz.begin();
            }

            typename std::vector<nonzero>::iterator end() {
                return nz.end();
            }

            ptrdiff_t next_nonzero() {
                int p = q.top();
                q.pop();
                return nz[p].col;
            }

            struct higher_than {
                value_type tol;
                higher_than(value_type tol) : tol(tol) {}

                bool operator()(const nonzero &v) const {
                    return fabs(v.val) > tol;
                }
            };

            struct L_first {
                ptrdiff_t dia;

                L_first(ptrdiff_t dia) : dia(dia) {}

                bool operator()(const nonzero &v) const {
                    return v.col < dia;
                }
            };

            struct by_abs_val {
                ptrdiff_t dia;

                by_abs_val(ptrdiff_t dia) : dia(dia) {}

                bool operator()(const nonzero &a, const nonzero &b) const {
                    if (a.col == dia) return true;
                    if (b.col == dia) return false;

                    return fabs(a.val) > fabs(b.val);
                }
            };

            struct by_col {
                bool operator()(const nonzero &a, const nonzero &b) const {
                    return a.col < b.col;
                }
            };

            void move_to(
                    int lp, int up, value_type tol,
                    build_matrix &LU, std::vector<ptrdiff_t> &uptr
                    )
            {
                typedef typename std::vector<nonzero>::iterator ptr;

                ptr b = nz.begin();
                ptr e = nz.end();

                // Move zeros to back:
                e = std::partition(b, e, higher_than(tol));

                // Split L and U:
                ptr m = std::partition(b, e, L_first(dia));

                // Get largest p elements in L and U.
                ptr lend = std::min(b + lp, m);
                ptr uend = std::min(m + up, e);

                if (lend != m) std::nth_element(b, lend, m, by_abs_val(dia));
                if (uend != e) std::nth_element(m, uend, e, by_abs_val(dia));

                // Sort entries by column number
                std::sort(b, lend, by_col());
                std::sort(m, uend, by_col());

                // copy L to the output matrix.
                for(ptr a = b; a != lend; ++a) {
                    LU.col.push_back(a->col);
                    LU.val.push_back(a->val);
                }

                // Store pointer to diagonal and invert its value.
                uptr.push_back(LU.val.size());
                m->val = 1 / m->val;

                // copy U to the output matrix.
                for(ptr a = m; a != uend; ++a) {
                    LU.col.push_back(a->col);
                    LU.val.push_back(a->val);
                }

                LU.ptr.push_back(LU.val.size());

                BOOST_FOREACH(const nonzero &e, nz) idx[e.col] = -1;
                nz.clear();
            }
        };

        template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
        void apply(
                const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp,
                const params &prm
                ) const
        {
            backend::residual(rhs, A, x, tmp);

            const size_t n = backend::rows(A);

            for(size_t i = 0; i < n; i++) {
                for(ptrdiff_t j = LU.ptr[i], e = uptr[i]; j < e; ++j)
                    tmp[i] -= LU.val[j] * tmp[LU.col[j]];
            }

            for(size_t i = n; i-- > 0;) {
                for(ptrdiff_t j = uptr[i] + 1, e = LU.ptr[i + 1]; j < e; ++j)
                    tmp[i] -= LU.val[j] * tmp[LU.col[j]];
                tmp[i] *= LU.val[uptr[i]];
            }

            backend::axpby(prm.damping, tmp, 1, x);
        }
};

} // namespace relaxation

namespace backend {

template <class Backend>
struct relaxation_is_supported<
    Backend,
    relaxation::ilut,
    typename boost::disable_if<
            typename boost::is_same<
                Backend,
                builtin<typename Backend::value_type>
            >::type
        >::type
    > : boost::false_type
{};

} // namespace backend
} // namespace amgcl

#endif
