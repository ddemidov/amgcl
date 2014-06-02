#ifndef AMGCL_BACKEND_BLOCK_CRS_HPP
#define AMGCL_BACKEND_BLOCK_CRS_HPP

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
 * \file   amgcl/backend/block_crs.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Sparse matrix in block-CRS format.
 */

#include <amgcl/backend/interface.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace backend {

template < typename V, typename C, typename P >
struct bcrs {
    typedef V val_type;
    typedef C col_type;
    typedef P ptr_type;

    size_t block_size;
    size_t nrows, ncols;
    size_t brows, bcols;

    std::vector<ptr_type> ptr;
    std::vector<col_type> col;
    std::vector<val_type> val;

    template < class Matrix >
    bcrs(const Matrix &A, size_t block_size)
        : block_size(block_size), nrows( rows(A) ), ncols( cols(A) ),
          brows((nrows + block_size - 1) / block_size),
          bcols((ncols + block_size - 1) / block_size),
          ptr(brows + 1, 0)
    {
#pragma omp parallel
        {
            std::vector<col_type> marker(bcols, static_cast<col_type>(-1));

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            size_t chunk_size  = (brows + nt - 1) / nt;
            size_t chunk_start = tid * chunk_size;
            size_t chunk_end   = std::min(brows, chunk_start + chunk_size);
#else
            size_t chunk_start = 0;
            size_t chunk_end   = brows;
#endif

            // Count number of nonzeros in block matrix.
            typedef typename backend::row_iterator<Matrix>::type row_iterator;
            for(size_t ib = chunk_start, ia = ib * block_size; ib < chunk_end; ++ib) {
                for(size_t k = 0; k < block_size && ia < nrows; ++k, ++ia) {
                    for(row_iterator a = backend::row_begin(A, ia); a; ++a) {
                        col_type cb = a.col() / block_size;

                        if (marker[cb] != static_cast<col_type>(ib)) {
                            marker[cb]  = static_cast<col_type>(ib);
                            ++ptr[ib + 1];
                        }
                    }
                }
            }

            std::fill(marker.begin(), marker.end(), static_cast<col_type>(-1));

#pragma omp barrier
#pragma omp single
            {
                std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
                col.resize(ptr.back());
                val.resize(ptr.back() * block_size * block_size, 0);
            }

            // Fill the block matrix.
            for(size_t ib = chunk_start, ia = ib * block_size; ib < chunk_end; ++ib) {
                ptr_type row_beg = ptr[ib];
                ptr_type row_end = row_beg;

                for(size_t k = 0; k < block_size && ia < nrows; ++k, ++ia) {
                    for(row_iterator a = backend::row_begin(A, ia); a; ++a) {
                        col_type cb = a.col() / block_size;
                        col_type cc = a.col() % block_size;
                        val_type va = a.value();

                        if (marker[cb] < row_beg) {
                            marker[cb] = row_end;
                            col[row_end] = cb;
                            val[block_size * (block_size * row_end + k) + cc] = va;
                            ++row_end;
                        } else {
                            val[block_size * (block_size * marker[cb] + k) + cc] = va;
                        }
                    }
                }
            }
        }
    }
};

//---------------------------------------------------------------------------
// Specialization of matrix interface
//---------------------------------------------------------------------------
template < typename V, typename C, typename P >
struct value_type< bcrs<V, C, P> > {
    typedef V type;
};

template < typename V, typename C, typename P >
struct rows_impl< bcrs<V, C, P> > {
    static size_t get(const bcrs<V, C, P> &A) {
        return A.nrows;
    }
};

template < typename V, typename C, typename P >
struct cols_impl< bcrs<V, C, P> > {
    static size_t get(const bcrs<V, C, P> &A) {
        return A.ncols;
    }
};

template < typename V, typename C, typename P >
struct nonzeros_impl< bcrs<V, C, P> > {
    static size_t get(const bcrs<V, C, P> &A) {
        return A.ptr.back() * A.block_size * A.block_size;
    }
};

//---------------------------------------------------------------------------
// block_crs backend definition
//---------------------------------------------------------------------------
template <typename real>
struct block_crs {
    typedef real value_type;
    typedef long index_type;

    typedef bcrs<real, index_type, index_type> matrix;
    typedef typename builtin<real>::vector     vector;

    struct params {
        size_t block_size;

        params(size_t block_size = 4) : block_size(block_size) {}
    };

    static boost::shared_ptr<matrix>
    copy_matrix(boost::shared_ptr< typename backend::builtin<real>::matrix > A,
            const params &prm)
    {
        return boost::make_shared<matrix>(*A, prm.block_size);
    }

    static boost::shared_ptr<vector>
    copy_vector(boost::shared_ptr< vector > x, const params&)
    {
        return x;
    }

    static boost::shared_ptr<vector>
    copy_vector(const vector &x, const params&)
    {
        return boost::make_shared<vector>(x);
    }

    static boost::shared_ptr<vector>
    create_vector(size_t size, const params&)
    {
        return boost::make_shared<vector>(size);
    }
};

template < typename V, typename C, typename P >
struct spmv_impl< bcrs<V, C, P>, std::vector<V> >
{
    typedef bcrs<V, C, P>  matrix;
    typedef std::vector<V> vector;

    static void apply(V alpha, const matrix &A, const vector &x,
            V beta, vector &y)
    {
        const size_t nb  = A.brows;
        const size_t na  = A.nrows;
        const size_t ma  = A.ncols;
        const size_t b1 = A.block_size;
        const size_t b2 = b1 * b1;

        if (beta) {
            if (beta != 1) {
#pragma omp parallel for
                for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(na); ++i) {
                    y[i] *= beta;
                }
            }
        } else {
            clear(y);
        }

#pragma omp parallel for
        for(ptrdiff_t ib = 0; ib < static_cast<ptrdiff_t>(nb); ++ib) {
            for(P jb = A.ptr[ib], eb = A.ptr[ib + 1]; jb < eb; ++jb) {
                const V *va = A.val.data() + jb * b2;
                size_t ia = ib * b1;
                size_t c0 = A.col[jb] * b1;
                for(size_t k = 0; k < b1 && ia < na; ++k, ++ia) {
                    V sum = 0;
                    size_t ca = c0;
                    for(size_t l = 0; l < b1; ++l, ++ca, ++va)
                        if (ca < ma) sum += (*va) * x[ca];
                    y[ia] += alpha * sum;
                }
            }
        }
    }
};

template < typename V, typename C, typename P >
struct residual_impl< bcrs<V, C, P>, std::vector<V> >
{
    typedef bcrs<V, C, P>  matrix;
    typedef std::vector<V> vector;

    static void apply(const vector &rhs, const matrix &A, const vector &x,
            vector &r)
    {
        copy(rhs, r);
        spmv(-1, A, x, 1, r);
    }
};

} // namespace backend
} // namespace amgcl

#endif
