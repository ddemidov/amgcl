#ifndef AMGCL_COARSENING_RUGE_STUBEN_HPP
#define AMGCL_COARSENING_RUGE_STUBEN_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/coarsening/ruge_stuben.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Ruge-Stuben coarsening with direct interpolation.
 */

#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/scaled_galerkin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace coarsening {

struct ruge_stuben {
    struct params {
        float eps_strong;
        float eps_trunc;
        bool  do_trunc;

        params() : eps_strong(0.25f), eps_trunc(0.2f), do_trunc(true) {}
    };

    template <typename Val, typename Col, typename Ptr>
    static boost::tuple<
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >,
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >
        >
    transfer_operators(
            const backend::crs<Val, Col, Ptr> &A,
            const params &prm)
    {
        typedef backend::crs<Val, Col, Ptr> matrix;
        const size_t n   = rows(A);

        std::vector<char> cf(n, 'U');
        backend::crs<char, Col, Ptr> S;

        TIC("C/F split");
        connect(A, prm.eps_strong, S, cf);
        cfsplit(A, S, cf);
        TOC("C/F split");

        TIC("interpolation");
        size_t nc = 0;
        std::vector<Ptr> cidx(n);
        for(size_t i = 0; i < n; ++i)
            if (cf[i] == 'C') cidx[i] = nc++;

        boost::shared_ptr<matrix> P = boost::make_shared<matrix>();
        P->nrows = n;
        P->ncols = nc;
        P->ptr.resize(n + 1, 0);

        std::vector<Val> Amin, Amax;

        if (prm.do_trunc) {
            Amin.resize(n);
            Amax.resize(n);
        }

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            if (cf[i] == 'C') {
                ++P->ptr[i + 1];
                continue;
            }

            if (prm.do_trunc) {
                Val amin = 0, amax = 0;

                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
                    if (!S.val[j] || cf[ A.col[j] ] != 'C') continue;

                    amin = std::min(amin, A.val[j]);
                    amax = std::max(amax, A.val[j]);
                }

                Amin[i] = (amin *= prm.eps_trunc);
                Amax[i] = (amax *= prm.eps_trunc);

                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
                    if (!S.val[j] || cf[A.col[j]] != 'C') continue;

                    if (A.val[j] <= amin || A.val[j] >= amax)
                        ++P->ptr[i + 1];
                }
            } else {
                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
                    if (S.val[j] && cf[A.col[j]] == 'C')
                        ++P->ptr[i + 1];
            }
        }

        boost::partial_sum(P->ptr, P->ptr.begin());
        P->col.resize(P->ptr.back());
        P->val.resize(P->ptr.back());

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            Ptr row_head = P->ptr[i];

            if (cf[i] == 'C') {
                P->col[row_head] = cidx[i];
                P->val[row_head] = 1;
                continue;
            }

            Val dia   = 0;
            Val a_num = 0, a_den = 0;
            Val b_num = 0, b_den = 0;
            Val d_neg = 0, d_pos = 0;

            for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
                Col c = A.col[j];
                Val v = A.val[j];

                if (c == i) {
                    dia = v;
                    continue;
                }

                if (v < 0) {
                    a_num += v;
                    if (S.val[j] && cf[c] == 'C') {
                        a_den += v;
                        if (prm.do_trunc && v > Amin[i]) d_neg += v;
                    }
                } else {
                    b_num += v;
                    if (S.val[j] && cf[c] == 'C') {
                        b_den += v;
                        if (prm.do_trunc && v < Amax[i]) d_pos += v;
                    }
                }
            }

            Val cf_neg = 1;
            Val cf_pos = 1;

            if (prm.do_trunc) {
                if (fabs(a_den - d_neg) > 1e-32) cf_neg = a_den / (a_den - d_neg);
                if (fabs(b_den - d_pos) > 1e-32) cf_pos = b_den / (b_den - d_pos);
            }

            if (b_num > 0 && fabs(b_den) < 1e-32) dia += b_num;

            Val alpha = fabs(a_den) > 1e-32 ? -cf_neg * a_num / (dia * a_den) : 0;
            Val beta  = fabs(b_den) > 1e-32 ? -cf_pos * b_num / (dia * b_den) : 0;

            for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
                Col c = A.col[j];
                Val v = A.val[j];

                if (!S.val[j] || cf[c] != 'C') continue;
                if (prm.do_trunc && v > Amin[i] && v < Amax[i]) continue;

                P->col[row_head] = cidx[c];
                P->val[row_head] = (v < 0 ? alpha : beta) * v;
                ++row_head;
            }
        }
        TOC("interpolation");

        boost::shared_ptr<matrix> R = boost::make_shared<matrix>();
        *R = transpose(*P);
        return boost::make_tuple(P, R);
    }

    template <typename Val, typename Col, typename Ptr>
    static boost::shared_ptr< backend::crs<Val, Col, Ptr> >
    coarse_operator(
            const backend::crs<Val, Col, Ptr> &A,
            const backend::crs<Val, Col, Ptr> &P,
            const backend::crs<Val, Col, Ptr> &R,
            const params&
            )
    {
        return detail::galerkin(A, P, R);
    }

    private:
        //-------------------------------------------------------------------
        // On return S will hold both strong connection matrix (in S.val, which
        // is piggybacking A.ptr and A.col), and its transposition (in S.ptr
        // and S.val).
        //
        // Variables that have no positive connections are marked as F(ine).
        //-------------------------------------------------------------------
        template <typename Val, typename Col, typename Ptr>
        static void connect(
                backend::crs<Val,  Col, Ptr> const &A, float eps_strong,
                backend::crs<char, Col, Ptr>       &S,
                std::vector<char>                  &cf
                )
        {
            typedef backend::crs<Val, Col, Ptr>                  matrix;
            typedef typename backend::row_iterator<matrix>::type row_iterator;

            const size_t n   = rows(A);
            const size_t nnz = nonzeros(A);

            S.nrows = S.ncols = n;
            S.ptr.resize( n+1 );
            S.val.resize( nnz );

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                Val a_min = 0;

                for(row_iterator a = row_begin(A, i); a; ++a)
                    if (a.col() != i) a_min = std::min(a_min, a.value());

                if (fabs(a_min) < 1e-32) {
                    cf[i] = 'F';
                    continue;
                }

                a_min *= eps_strong;

                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
                    S.val[j] = (A.col[j] != i && A.val[j] < a_min);
            }

            // Transposition of S:
            for(size_t i = 0; i < nnz; ++i)
                if (S.val[i]) ++( S.ptr[ A.col[i] + 1] );

            boost::partial_sum(S.ptr, S.ptr.begin());

            S.col.resize( S.ptr.back() );

            for(size_t i = 0; i < n; ++i)
                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
                    if (S.val[j]) S.col[ S.ptr[ A.col[j] ]++ ] = i;

            std::rotate(S.ptr.begin(), S.ptr.end() - 1, S.ptr.end());
            S.ptr.front() = 0;
        }

        // Split variables into C(oarse) and F(ine) sets.
        template <typename Val, typename Col, typename Ptr>
        static void cfsplit(
                backend::crs<Val,  Col, Ptr> const &A,
                backend::crs<char, Col, Ptr> const &S,
                std::vector<char>                  &cf
                )
        {
            const size_t n = rows(A);

            std::vector<Col> lambda(n);

            // Initialize lambdas:
            for(size_t i = 0; i < n; ++i) {
                Col temp = 0;
                for(Ptr j = S.ptr[i], e = S.ptr[i+1]; j < e; ++j)
                    temp += ( cf[ S.col[j] ] == 'U' ? 1 : 2 );
                lambda[i] = temp;
            }

            // Keep track of variable groups with equal lambda values.
            // ptr - start of a group;
            // cnt - size of a group;
            // i2n - variable number;
            // n2i - vaiable position in a group.
            std::vector<Ptr> ptr(n+1, 0);
            std::vector<Ptr> cnt(n, 0);
            std::vector<Ptr> i2n(n);
            std::vector<Ptr> n2i(n);

            for(size_t i = 0; i < n; ++i) ++ptr[lambda[i] + 1];

            boost::partial_sum(ptr, ptr.begin());

            for(size_t i = 0; i < n; ++i) {
                Col lam = lambda[i];
                Ptr idx = ptr[lam] + cnt[lam]++;
                i2n[idx] = i;
                n2i[i] = idx;
            }

            // Process variables by decreasing lambda value.
            // 1. The vaiable with maximum value of lambda becomes next C-variable.
            // 2. Its neighbours from S' become F-variables.
            // 3. Keep lambda values in sync.
            for(size_t top = n; top-- > 0; ) {
                Ptr i   = i2n[top];
                Col lam = lambda[i];

                if (lam == 0) {
                    boost::replace(cf, 'U', 'C');
                    break;
                }

                // Remove tne variable from its group.
                --cnt[lam];

                if (cf[i] == 'F') continue;

                // Mark the variable as 'C'.
                cf[i] = 'C';

                // Its neighbours from S' become F-variables.
                for(Ptr j = S.ptr[i], e = S.ptr[i + 1]; j < e; ++j) {
                    Col c = S.col[j];

                    if (cf[c] != 'U') continue;

                    cf[c] = 'F';

                    // Increase lambdas of the newly created F's neighbours.
                    for(Ptr aj = A.ptr[c], ae = A.ptr[c + 1]; aj < ae; ++aj) {
                        if (!S.val[aj]) continue;

                        Col ac    = A.col[aj];
                        Col lam_a = lambda[ac];

                        if (cf[ac] != 'U' || static_cast<size_t>(lam_a) + 1 >= n)
                            continue;

                        Ptr old_pos = n2i[ac];
                        Ptr new_pos = ptr[lam_a] + cnt[lam_a] - 1;

                        n2i[i2n[old_pos]] = new_pos;
                        n2i[i2n[new_pos]] = old_pos;

                        std::swap(i2n[old_pos], i2n[new_pos]);

                        --cnt[lam_a];
                        ++cnt[lam_a + 1];
                        ptr[lam_a + 1] = ptr[lam_a] + cnt[lam_a];

                        lambda[ac] = lam_a + 1;
                    }
                }

                // Decrease lambdas of the newly create C's neighbours.
                for(Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; j++) {
                    if (!S.val[j]) continue;

                    Col c   = A.col[j];
                    Col lam = lambda[c];

                    if (cf[c] != 'U' || lam == 0) continue;

                    Ptr old_pos = n2i[c];
                    Ptr new_pos = ptr[lam];

                    n2i[i2n[old_pos]] = new_pos;
                    n2i[i2n[new_pos]] = old_pos;

                    std::swap(i2n[old_pos], i2n[new_pos]);

                    --cnt[lam];
                    ++cnt[lam - 1];
                    ++ptr[lam];
                    lambda[c] = lam - 1;
                }
            }
        }
};

} // namespace coarsening
} // namespace amgcl

#endif
