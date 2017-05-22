#ifndef AMGCL_RELAXATION_VEXCL_ILU_HPP
#define AMGCL_RELAXATION_VEXCL_ILU_HPP

/*
The MIT License

Copyright (c) 2012-2017 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/relaxation/vexcl_ilu.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  VexCL-spacific implementation of solver for sparse triangular
 *         systems obtained as a result of an incomplete LU factorization.
 */

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
#include <amgcl/relaxation/detail/ilu_solve.hpp>

namespace amgcl {
namespace relaxation {
namespace detail {

template <class value_type, class DS>
class ilu_solve< backend::vexcl<value_type, DS> > {
    public:
        typedef backend::vexcl<value_type, DS> Backend;
        typedef typename Backend::matrix matrix;
        typedef typename Backend::vector vector;
        typedef typename Backend::matrix_diagonal matrix_diagonal;
        typedef typename backend::builtin<value_type>::matrix build_matrix;
        typedef typename Backend::rhs_type rhs_type;
        typedef typename math::scalar_of<value_type>::type scalar_type;

        template <class Params>
        ilu_solve(
                boost::shared_ptr<build_matrix> L,
                boost::shared_ptr<build_matrix> U,
                boost::shared_ptr<backend::numa_vector<value_type> > D,
                const Params &, const typename Backend::params &bprm
                ) : lower(bprm.q, *L, D->data()), upper(bprm.q, *U, D->data())
        { }

        template <class Vector>
        void solve(Vector &x) {
            lower.solve(x);
            upper.solve(x);
        }

    private:
        template <bool lower>
        struct sptr_solve {
            ptrdiff_t nlev;

            std::vector<matrix>                 L;
            std::vector<vex::vector<ptrdiff_t>> I;
            std::vector<matrix_diagonal>        D;

            template <class Matrix>
            sptr_solve(const std::vector<vex::backend::command_queue> &q,
                    const Matrix &A, const value_type *_D = 0) : nlev(0)
            {
                ptrdiff_t n = A.nrows;

                std::vector<ptrdiff_t> level(n, 0);
                std::vector<ptrdiff_t> order(n, 0);


                // 1. split rows into levels.
                ptrdiff_t beg = lower ? 0 : n-1;
                ptrdiff_t end = lower ? n :  -1;
                ptrdiff_t inc = lower ? 1 :  -1;

                for(ptrdiff_t i = beg; i != end; i += inc) {
                    ptrdiff_t l = level[i];

                    for(ptrdiff_t j = A.ptr[i]; j < A.ptr[i+1]; ++j)
                        l = std::max(l, level[A.col[j]]+1);

                    level[i] = l;
                    nlev = std::max(nlev, l+1);
                }


                // 2. reorder matrix rows.
                std::vector<ptrdiff_t> start(nlev+1, 0);

                for(ptrdiff_t i = 0; i < n; ++i)
                    ++start[level[i]+1];

                std::partial_sum(start.begin(), start.end(), start.begin());

                for(ptrdiff_t i = 0; i < n; ++i)
                    order[start[level[i]]++] = i;

                std::rotate(start.begin(), start.end() - 1, start.end());
                start[0] = 0;


                // 3. Create levels.
                L.reserve(nlev);
                I.reserve(nlev);
                if(!lower) D.reserve(nlev);

                for(ptrdiff_t lev = 0; lev < nlev; ++lev) {
                    // split each level into tasks.
                    ptrdiff_t rows = start[lev+1] - start[lev];

                    std::vector<ptrdiff_t>  ptr(rows + 1); ptr[0] = 0;
                    std::vector<ptrdiff_t>  ord(rows);
                    std::vector<value_type> dia; if (!lower) dia.resize(rows);

                    // count nonzeros in the current level
                    for(ptrdiff_t i = start[lev], k = 0; i < start[lev+1]; ++i, ++k) {
                        ptrdiff_t j = order[i];
                        ptr[k+1] = ptr[k] + A.ptr[j+1] - A.ptr[j];
                        ord[k] = j;
                        if (!lower) dia[k] = _D[j];
                    }

                    std::vector<ptrdiff_t>    col(ptr[rows]);
                    std::vector<value_type> val(ptr[rows]);

                    // copy nonzeros
                    for(ptrdiff_t i = start[lev], k = 0; i < start[lev+1]; ++i, ++k) {
                        ptrdiff_t o = ord[k];
                        ptrdiff_t h = ptr[k];
                        for(int j = A.ptr[o]; j < A.ptr[o+1]; ++j) {
                            col[h] = A.col[j];
                            val[h] = A.val[j];
                            ++h;
                        }
                    }

                    L.emplace_back(q, rows, n, ptr, col, val);
                    I.emplace_back(q, ord);
                    if (!lower) D.emplace_back(q, dia);
                }
            }

            template <class Vector>
            void solve(Vector &x) const {
                do_solve(x, is_static_matrix<value_type>());
            }

            template <class Vector>
            void do_solve(Vector &x, boost::false_type) const {
                for(ptrdiff_t i = 0; i < nlev; ++i) {
                    using namespace vex;

                    auto _x = tag<1>(x);
                    auto _I = tag<2>(I[i]);
                    auto _y = permutation(_I)(_x);

                    if (lower) {
                        _y -= L[i] * _x;
                    } else {
                        _y = D[i] * (_y - L[i] * _x);
                    }
                }
            }

            template <class Vector>
            void do_solve(Vector &x, boost::true_type) const {
                using backend::vex_sub;
                using backend::vex_mul;

                typedef typename math::scalar_of<value_type>::type T;
                const int B = math::static_rows<value_type>::value;

                for(ptrdiff_t i = 0; i < nlev; ++i) {
                    using namespace vex;

                    auto _x = tag<1>(x);
                    auto _I = tag<2>(I[i]);
                    auto _y = permutation(_I)(_x);

                    if (lower) {
                        _y = vex_sub<T,B>().apply(_y, L[i] * _x);
                    } else {
                        _y = vex_mul<T,B>().apply(D[i], vex_sub<T,B>().apply(_y, L[i] * _x));
                    }
                }
            }

        };

        sptr_solve<true>  lower;
        sptr_solve<false> upper;
};

} // namespace detail
} // namespace relaxation
} // namespace amgcl

#endif
