#ifndef AMGCL_DETAIL_SPGEMM_HPP
#define AMGCL_DETAIL_SPGEMM_HPP

/*
The MIT License

Copyright (c) 2012-2019 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/detail/spgemm.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Sparse matrix-matrix product algorithms.
 *
 * This implements two algorithms.
 *
 * The first is an OpenMP-enabled modification of classic algorithm from Saad
 * [1]. It is used whenever number of OpenMP cores is 4 or less.
 *
 * The second is Row-merge algorithm from Rupp et al. [2]. The algorithm
 * requires less memory and shows much better scalability than classic one.
 * It is used when number of OpenMP cores is more than 4.
 *
 * [1] Saad, Yousef. Iterative methods for sparse linear systems. Siam, 2003.
 * [2] Rupp K, Rudolf F, Weinbub J, Morhammer A, Grasser T, Jungel A. Optimized
 *     Sparse Matrix-Matrix Multiplication for Multi-Core CPUs, GPUs, and Xeon
 *     Phi. Submitted
 */
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/detail/sort_row.hpp>

namespace amgcl {
namespace backend {

//---------------------------------------------------------------------------
template <class AMatrix, class BMatrix, class CMatrix>
void spgemm_saad(const AMatrix &A, const BMatrix &B, CMatrix &C, bool sort = true)
{
    typedef typename backend::value_type<CMatrix>::type Val;
    typedef ptrdiff_t Idx;

    auto Aptr = backend::ptr_data(A);
    auto Acol = backend::col_data(A);
    auto Aval = backend::val_data(A);

    auto Bptr = backend::ptr_data(B);
    auto Bcol = backend::col_data(B);
    auto Bval = backend::val_data(B);

    C.set_size(backend::rows(A), backend::cols(B));
    C.ptr[0] = 0;

#pragma omp parallel
    {
        std::vector<ptrdiff_t> marker(C.ncols, -1);

#pragma omp for
        for(Idx ia = 0; ia < static_cast<Idx>(C.nrows); ++ia) {
            Idx C_cols = 0;
            for(Idx ja = Aptr[ia], ea = Aptr[ia+1]; ja < ea; ++ja) {
                Idx ca = Acol[ja];

                for(Idx jb = Bptr[ca], eb = Bptr[ca+1]; jb < eb; ++jb) {
                    Idx cb = Bcol[jb];
                    if (marker[cb] != ia) {
                        marker[cb]  = ia;
                        ++C_cols;
                    }
                }
            }
            C.ptr[ia + 1] = C_cols;
        }
    }

    C.set_nonzeros(C.scan_row_sizes());

#pragma omp parallel
    {
        std::vector<ptrdiff_t> marker(C.ncols, -1);

#pragma omp for
        for(Idx ia = 0; ia < static_cast<Idx>(C.nrows); ++ia) {
            Idx row_beg = C.ptr[ia];
            Idx row_end = row_beg;

            for(Idx ja = Aptr[ia], ea = Aptr[ia+1]; ja < ea; ++ja) {
                Idx ca = Acol[ja];
                Val va = Aval[ja];

                for(Idx jb = Bptr[ca], eb = Bptr[ca+1]; jb < eb; ++jb) {
                    Idx cb = Bcol[jb];
                    Val vb = Bval[jb];

                    if (marker[cb] < row_beg) {
                        marker[cb] = row_end;
                        C.col[row_end] = cb;
                        C.val[row_end] = va * vb;
                        ++row_end;
                    } else {
                        C.val[marker[cb]] += va * vb;
                    }
                }
            }

            if (sort) amgcl::detail::sort_row(
                    C.col + row_beg, C.val + row_beg, row_end - row_beg);
        }
    }
}

//---------------------------------------------------------------------------
template <bool need_out, class Idx>
Idx* merge_rows(
        const Idx *col1, const Idx *col1_end,
        const Idx *col2, const Idx *col2_end,
        Idx *col3
        )
{
    while(col1 != col1_end && col2 != col2_end) {
        Idx c1 = *col1;
        Idx c2 = *col2;

        if (c1 < c2) {
            if (need_out) *col3 = c1;
            ++col1;
        } else if (c1 == c2) {
            if (need_out) *col3 = c1;
            ++col1;
            ++col2;
        } else {
            if (need_out) *col3 = c2;
            ++col2;
        }
        ++col3;
    }

    if (need_out) {
        if (col1 < col1_end) {
            return std::copy(col1, col1_end, col3);
        } else if (col2 < col2_end) {
            return std::copy(col2, col2_end, col3);
        } else {
            return col3;
        }
    } else {
        return col3 + (col1_end - col1) + (col2_end - col2);
    }
}

template <class A1, class Col1, class Val1, class A2, class Col2, class Val2, class Col, class Val>
Col* merge_rows(
        const A1 &alpha1, Col1 col1, Col1 col1_end, Val1 val1,
        const A2 &alpha2, Col2 col2, Col2 col2_end, Val2 val2,
        Col *col3, Val *val3
        )
{
    while(col1 != col1_end && col2 != col2_end) {
        auto c1 = *col1;
        auto c2 = *col2;

        if (c1 < c2) {
            ++col1;

            *col3 = c1;
            *val3 = alpha1 * (*val1++);
        } else if (c1 == c2) {
            ++col1;
            ++col2;

            *col3 = c1;
            *val3 = alpha1 * (*val1++) + alpha2 * (*val2++);
        } else {
            ++col2;

            *col3 = c2;
            *val3 = alpha2 * (*val2++);
        }

        ++col3;
        ++val3;
    }

    while(col1 < col1_end) {
        *col3++ = *col1++;
        *val3++ = alpha1 * (*val1++);
    }

    while(col2 < col2_end) {
        *col3++ = *col2++;
        *val3++ = alpha2 * (*val2++);
    }

    return col3;
}

template <class ACol, class BPtr, class BCol, class TCol>
ptrdiff_t prod_row_width(
        ACol acol, ACol acol_end,
        BPtr bptr, BCol bcol,
        TCol tmp_col1, TCol tmp_col2, TCol tmp_col3
        )
{
    const auto nrows = acol_end - acol;

    /* No rows to merge, nothing to do */
    if (nrows == 0) return 0;

    /* Single row, just copy it to output */
    if (nrows == 1) return bptr[*acol + 1] - bptr[*acol];

    /* Two rows, merge them */
    if (nrows == 2) {
        auto a1 = acol[0];
        auto a2 = acol[1];

        return merge_rows<false>(
                bcol + bptr[a1], bcol + bptr[a1+1],
                bcol + bptr[a2], bcol + bptr[a2+1],
                tmp_col1
                ) - tmp_col1;
    }

    /* Generic case (more than two rows).
     *
     * Merge rows by pairs, then merge the results together.
     * When merging two rows, the result is always wider (or equal).
     * Merging by pairs allows to work with short rows as often as possible.
     */
    // Merge first two.
    auto a1 = *acol++;
    auto a2 = *acol++;
    auto c_col1 = merge_rows<true>(
            bcol + bptr[a1], bcol + bptr[a1+1],
            bcol + bptr[a2], bcol + bptr[a2+1],
            tmp_col1
            ) - tmp_col1;

    // Go by pairs.
    while(acol + 1 < acol_end) {
        a1 = *acol++;
        a2 = *acol++;

        auto c_col2 = merge_rows<true>(
                bcol + bptr[a1], bcol + bptr[a1+1],
                bcol + bptr[a2], bcol + bptr[a2+1],
                tmp_col2
                ) - tmp_col2;

        if (acol == acol_end) {
            return merge_rows<false>(
                    tmp_col1, tmp_col1 + c_col1,
                    tmp_col2, tmp_col2 + c_col2,
                    tmp_col3
                    ) - tmp_col3;
        } else {
            c_col1 = merge_rows<true>(
                    tmp_col1, tmp_col1 + c_col1,
                    tmp_col2, tmp_col2 + c_col2,
                    tmp_col3
                    ) - tmp_col3;

            std::swap(tmp_col1, tmp_col3);
        }
    }

    // Merge the tail.
    a2 = *acol;
    return merge_rows<false>(
            tmp_col1, tmp_col1 + c_col1,
            bcol + bptr[a2], bcol + bptr[a2+1],
            tmp_col2
            ) - tmp_col2;
}

template <class ACol, class AVal, class BPtr, class BCol, class BVal, class CCol, class CVal, class TCol, class TVal>
void prod_row(
        ACol acol, ACol acol_end, AVal aval,
        BPtr bptr, BCol bcol, BVal bval,
        CCol out_col, CVal out_val,
        TCol tm2_col, TVal tm2_val,
        TCol tm3_col, TVal tm3_val
        )
{
    typedef typename std::decay<decltype(*out_val)>::type Val;

    const auto nrows = acol_end - acol;

    /* No rows to merge, nothing to do */
    if (nrows == 0) return;

    /* Single row, just copy it to output */
    if (nrows == 1) {
        auto ac = *acol;
        auto av = *aval;

        auto bv = bval + bptr[ac];
        auto bc = bcol + bptr[ac];
        auto be = bcol + bptr[ac+1];

        while(bc != be) {
            *out_col++ = *bc++;
            *out_val++ = av * (*bv++);
        }

        return;
    }

    /* Two rows, merge them */
    if (nrows == 2) {
        auto ac1 = acol[0];
        auto ac2 = acol[1];

        auto av1 = aval[0];
        auto av2 = aval[1];

        merge_rows(
                av1, bcol + bptr[ac1], bcol + bptr[ac1+1], bval + bptr[ac1],
                av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2],
                out_col, out_val
                );

        return;
    }

    /* Generic case (more than two rows).
     *
     * Merge rows by pairs, then merge the results together.
     * When merging two rows, the result is always wider (or equal).
     * Merging by pairs allows to work with short rows as often as possible.
     */
    // Merge first two.
    auto ac1 = *acol++;
    auto ac2 = *acol++;

    auto av1 = *aval++;
    auto av2 = *aval++;

    auto *tm1_col = out_col;
    auto *tm1_val = out_val;

    auto c_col1 = merge_rows(
            av1, bcol + bptr[ac1], bcol + bptr[ac1+1], bval + bptr[ac1],
            av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2],
            tm1_col, tm1_val
            ) - tm1_col;

    // Go by pairs.
    while(acol + 1 < acol_end) {
        ac1 = *acol++;
        ac2 = *acol++;

        av1 = *aval++;
        av2 = *aval++;

        auto c_col2 = merge_rows(
                av1, bcol + bptr[ac1], bcol + bptr[ac1+1], bval + bptr[ac1],
                av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2],
                tm2_col, tm2_val
                ) - tm2_col;

        c_col1 = merge_rows(
                math::identity<Val>(), tm1_col, tm1_col + c_col1, tm1_val,
                math::identity<Val>(), tm2_col, tm2_col + c_col2, tm2_val,
                tm3_col, tm3_val
                ) - tm3_col;

        std::swap(tm3_col, tm1_col);
        std::swap(tm3_val, tm1_val);
    }

    // Merge the tail if there is one.
    if (acol < acol_end) {
        ac2 = *acol++;
        av2 = *aval++;

        c_col1 = merge_rows(
                math::identity<Val>(), tm1_col, tm1_col + c_col1, tm1_val,
                av2, bcol + bptr[ac2], bcol + bptr[ac2+1], bval + bptr[ac2],
                tm3_col, tm3_val
                ) - tm3_col;

        std::swap(tm3_col, tm1_col);
        std::swap(tm3_val, tm1_val);
    }

    // If we are lucky, tm1 now points to out.
    // Otherwise, copy the results.
    if (tm1_col != out_col) {
        std::copy(tm1_col, tm1_col + c_col1, out_col);
        std::copy(tm1_val, tm1_val + c_col1, out_val);
    }
}

template <class AMatrix, class BMatrix, class CMatrix>
void spgemm_rmerge(const AMatrix &A, const BMatrix &B, CMatrix &C) {
    typedef typename backend::value_type<CMatrix>::type Val;
    typedef ptrdiff_t Idx;

    Idx max_row_width = 0;

    auto Aptr = backend::ptr_data(A);
    auto Acol = backend::col_data(A);
    auto Aval = backend::val_data(A);

    auto Bptr = backend::ptr_data(B);
    auto Bcol = backend::col_data(B);
    auto Bval = backend::val_data(B);

    C.set_size(backend::rows(A), backend::cols(B));
    C.ptr[0] = 0;

#pragma omp parallel
    {
        Idx my_max = 0;

#pragma omp for
        for(int i = 0; i < static_cast<Idx>(C.nrows); ++i) {
            Idx row_beg = Aptr[i];
            Idx row_end = Aptr[i+1];
            Idx row_width = 0;
            for(Idx j = row_beg; j < row_end; ++j) {
                Idx a_col = Acol[j];
                row_width += Bptr[a_col + 1] - Bptr[a_col];
            }
            my_max = std::max(my_max, row_width);
        }

#pragma omp critical
        max_row_width = std::max(max_row_width, my_max);
    }

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif

    std::vector< std::vector<Idx> > tmp_col(nthreads);
    std::vector< std::vector<Val> > tmp_val(nthreads);

    for(int i = 0; i < nthreads; ++i) {
        tmp_col[i].resize(3 * max_row_width);
        tmp_val[i].resize(2 * max_row_width);
    }

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif

        Idx *t_col = &tmp_col[tid][0];

#pragma omp for
        for(Idx i = 0; i < static_cast<Idx>(C.nrows); ++i) {
            Idx row_beg = Aptr[i];
            Idx row_end = Aptr[i+1];

            C.ptr[i+1] = prod_row_width(
                    Acol + row_beg, Acol + row_end, Bptr, Bcol,
                    t_col, t_col + max_row_width, t_col + 2 * max_row_width
                    );
        }
    }

    C.set_nonzeros(C.scan_row_sizes());

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif

        Idx *t_col = tmp_col[tid].data();
        Val *t_val = tmp_val[tid].data();

#pragma omp for
        for(Idx i = 0; i < static_cast<Idx>(C.nrows); ++i) {
            Idx row_beg = Aptr[i];
            Idx row_end = Aptr[i+1];

            prod_row(
                    Acol + row_beg, Acol + row_end, Aval + row_beg,
                    Bptr, Bcol, Bval,
                    C.col + C.ptr[i], C.val + C.ptr[i],
                    t_col, t_val, t_col + max_row_width, t_val + max_row_width
                    );
        }
    }
}

} // namespace backend
} // namespace amgcl

#endif
