#ifndef AMGCL_BACKEND_BUILTIN_HPP
#define AMGCL_BACKEND_BUILTIN_HPP

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
 * \file   amgcl/backend/builtin.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Builtin backend.
 */

#include <vector>
#include <numeric>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <boost/range.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <amgcl/util.hpp>
#include <amgcl/backend/interface.hpp>

namespace amgcl {
namespace backend {

/// Sparse matrix stored in CRS format.
template <
    typename val_t = double,
    typename col_t = int,
    typename ptr_t = col_t
    >
struct crs {
    typedef val_t val_type;
    typedef col_t col_type;
    typedef ptr_t ptr_type;

    size_t nrows, ncols;
    std::vector<ptr_type> ptr;
    std::vector<col_type> col;
    std::vector<val_type> val;

    crs() : nrows(0), ncols(0) {}

    template <
        class PtrRange,
        class ColRange,
        class ValRange
        >
    crs(size_t nrows, size_t ncols,
        const PtrRange &ptr_range,
        const ColRange &col_range,
        const ValRange &val_range
        )
    : nrows(nrows), ncols(ncols),
      ptr(boost::begin(ptr_range), boost::end(ptr_range)),
      col(boost::begin(col_range), boost::end(col_range)),
      val(boost::begin(val_range), boost::end(val_range))
    {
        precondition(
                ptr.size() == nrows + 1                       &&
                static_cast<size_t>(ptr.back()) == col.size() &&
                static_cast<size_t>(ptr.back()) == val.size(),
                "Inconsistent sizes in crs constructor"
                );
    }

    template <class Matrix>
    crs(const Matrix &A) : nrows(backend::rows(A)), ncols(backend::cols(A))
    {
        ptr.reserve(nrows + 1);
        ptr.push_back(0);

        col.reserve(backend::nonzeros(A));
        val.reserve(backend::nonzeros(A));

        typedef typename backend::row_iterator<Matrix>::type row_iterator;
        for(size_t i = 0; i < nrows; ++i) {
            for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                col.push_back(a.col());
                val.push_back(a.value());
            }
            ptr.push_back(col.size());
        }
    }

    class row_iterator {
        public:
            operator bool() const {
                return m_col < m_end;
            }

            row_iterator& operator++() {
                ++m_col;
                ++m_val;
                return *this;
            }

            col_type col() const {
                return *m_col;
            }

            val_type value() const {
                return *m_val;
            }

        private:
            friend struct crs;

            const col_type * m_col;
            const col_type * m_end;
            const val_type * m_val;

            row_iterator(
                    const col_type * col,
                    const col_type * end,
                    const val_type * val
                    ) : m_col(col), m_end(end), m_val(val)
            {}
    };

    row_iterator row_begin(size_t row) const {
        ptr_type p = ptr[row];
        ptr_type e = ptr[row + 1];
        return row_iterator(&col[p], &col[e], &val[p]);
    }

    //-----------------------------------------------------------------------
    // Transpose of a sparse matrix.
    //-----------------------------------------------------------------------
    friend crs transpose(const crs &A)
    {
        const size_t n   = rows(A);
        const size_t m   = cols(A);
        const size_t nnz = nonzeros(A);

        crs T;
        T.nrows = m;
        T.ncols = n;
        T.ptr.resize(m+1);
        T.col.resize(nnz);
        T.val.resize(nnz);

        std::fill(T.ptr.begin(), T.ptr.end(), ptr_type());

        for(size_t j = 0; j < nnz; ++j)
            ++( T.ptr[A.col[j] + 1] );

        std::partial_sum(T.ptr.begin(), T.ptr.end(), T.ptr.begin());

        for(size_t i = 0; i < n; i++) {
            for(ptr_type j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
                ptr_type head = T.ptr[A.col[j]]++;

                T.col[head] = i;
                T.val[head] = A.val[j];
            }
        }

        std::rotate(T.ptr.begin(), T.ptr.end() - 1, T.ptr.end());
        T.ptr.front() = 0;

        return T;
    }

    //-----------------------------------------------------------------------
    // Matrix-matrix product.
    //-----------------------------------------------------------------------
    friend crs product(const crs &A, const crs &B) {
        const size_t n = rows(A);
        const size_t m = cols(B);

        crs C;
        C.nrows = n;
        C.ncols = m;
        C.ptr.resize(n + 1);
        std::fill(C.ptr.begin(), C.ptr.end(), ptr_type());

#pragma omp parallel
        {
            std::vector<col_type> marker(m, static_cast<col_type>(-1));

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            size_t chunk_size  = (n + nt - 1) / nt;
            size_t chunk_start = tid * chunk_size;
            size_t chunk_end   = std::min(n, chunk_start + chunk_size);
#else
            size_t chunk_start = 0;
            size_t chunk_end   = n;
#endif

            for(size_t ia = chunk_start; ia < chunk_end; ++ia) {
                for(row_iterator a = A.row_begin(ia); a; ++a) {
                    for(row_iterator b = B.row_begin(a.col()); b; ++b) {
                        if (marker[b.col()] != static_cast<col_type>(ia)) {
                            marker[b.col()]  = static_cast<col_type>(ia);
                            ++( C.ptr[ia + 1] );
                        }
                    }
                }
            }

            std::fill(marker.begin(), marker.end(), static_cast<col_type>(-1));

#pragma omp barrier
#pragma omp single
            {
                std::partial_sum(C.ptr.begin(), C.ptr.end(), C.ptr.begin());
                C.col.resize(C.ptr.back());
                C.val.resize(C.ptr.back());
            }

            for(size_t ia = chunk_start; ia < chunk_end; ++ia) {
                ptr_type row_beg = C.ptr[ia];
                ptr_type row_end = row_beg;

                for(row_iterator a = A.row_begin(ia); a; ++a) {
                    col_type ca = a.col();
                    val_type va = a.value();

                    for(row_iterator b = B.row_begin(ca); b; ++b) {
                        col_type cb = b.col();
                        val_type vb = b.value();

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
            }
        }

        return C;
    }

    //-----------------------------------------------------------------------
    // Diagonal entries of a sparse matrix.
    //-----------------------------------------------------------------------
    friend std::vector<val_type> diagonal(const crs &A, bool invert = false)
    {
        const size_t n = rows(A);
        std::vector<val_type> dia(n);

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            for(row_iterator a = A.row_begin(i); a; ++a) {
                if (a.col() == i) {
                    dia[i] = invert ? 1 / a.value() : a.value();
                    break;
                }
            }
        }

        return dia;
    }

    // Sort rows of the matrix column-wise.
    friend void sort_rows(crs &A) {
        const size_t n = rows(A);

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            ptr_type beg = A.ptr[i];
            ptr_type end = A.ptr[i + 1];
            insertion_sort(A.col.data() + beg, A.val.data() + beg, end - beg);
        }
    }

    static void insertion_sort(col_type *col, val_type *val, int n) {
        for(int j = 1; j < n; ++j) {
            col_type c = col[j];
            val_type v = val[j];
            int i = j - 1;
            while(i >= 0 && col[i] > c) {
                col[i + 1] = col[i];
                val[i + 1] = val[i];
                i--;
            }
            col[i + 1] = c;
            val[i + 1] = v;
        }
    }

    static void gaussj(int n, val_type *a) {
        std::vector<int>  idxc(n);
        std::vector<int>  idxr(n);
        std::vector<char> ipiv(n, false);

        for(int i = 0; i < n; ++i) {
            int irow = 0, icol = 0;

            val_type big = 0;
            for(int j = 0; j < n; ++j) {
                if (ipiv[j]) continue;

                for(int k = 0; k < n; ++k) {
                    if (!ipiv[k] && fabs(a[j * n + k]) > big) {
                        big  = fabs(a[j * n + k]);
                        irow = j;
                        icol = k;
                    }
                }
            }

            ipiv[icol] = true;

            if (irow != icol)
                std::swap_ranges(
                        a + n * irow, a + n * (irow + 1),
                        a + n * icol
                        );

            idxr[i] = irow;
            idxc[i] = icol;

            if (a[icol * n + icol] == 0)
                throw std::logic_error("Singular matrix in gaussj");

            val_type pivinv = 1 / a[icol * n + icol];
            a[icol * n + icol] = 1;

            for(val_type *v = a + icol * n, *e = a + (icol + 1) * n; v != e; ++v)
                *v *= pivinv;

            for(int k = 0; k < n; ++k) {
                if (k != icol) {
                    val_type dum = a[k * n + icol];
                    a[k * n + icol] = 0;
                    for(val_type *v1 = a + n * k, *v2 = a + n * icol, *e = a + n * (k + 1); v1 != e; ++v1, ++v2)
                        *v1 -= *v2 * dum;
                }
            }
        }

        for(int i = n - 1; i >= 0; --i) {
            if (idxr[i] != idxc[i]) {
                for(int j = 0; j < n; ++j)
                    std::swap(a[j * n + idxr[i]], a[j * n + idxc[i]]);
            }
        }
    }

    friend crs inverse(const crs &A) {
        const size_t n = rows(A);

        crs Ainv;
        Ainv.nrows = n;
        Ainv.ncols = n;
        Ainv.ptr.resize(n + 1);
        Ainv.col.resize(n * n);
        Ainv.val.resize(n * n);

        std::fill(Ainv.val.begin(), Ainv.val.end(), static_cast<val_type>(0));

        for(size_t i = 0; i < n; ++i)
            for(row_iterator a = A.row_begin(i); a; ++a)
                Ainv.val[i * n + a.col()] = a.value();

        gaussj(n, Ainv.val.data());

        Ainv.ptr[0] = 0;
        for(size_t i = 0, idx = 0; i < n; ) {
            for(size_t j = 0; j < n; ++j, ++idx) Ainv.col[idx] = j;

            Ainv.ptr[++i] = idx;
        }

        return Ainv;
    }
};

//---------------------------------------------------------------------------
// Specialization of matrix interface
//---------------------------------------------------------------------------
template < typename V, typename C, typename P >
struct value_type< crs<V, C, P> > {
    typedef V type;
};

template < typename V, typename C, typename P >
struct rows_impl< crs<V, C, P> > {
    static size_t get(const crs<V, C, P> &A) {
        return A.nrows;
    }
};

template < typename V, typename C, typename P >
struct cols_impl< crs<V, C, P> > {
    static size_t get(const crs<V, C, P> &A) {
        return A.ncols;
    }
};

template < typename V, typename C, typename P >
struct nonzeros_impl< crs<V, C, P> > {
    static size_t get(const crs<V, C, P> &A) {
        return A.ptr.empty() ? 0 : A.ptr.back();
    }
};

template < typename V, typename C, typename P >
struct row_iterator< crs<V, C, P> > {
    typedef
        typename crs<V, C, P>::row_iterator
        type;
};

template < typename V, typename C, typename P >
struct row_begin_impl< crs<V, C, P> > {
    typedef crs<V, C, P> Matrix;
    static typename row_iterator<Matrix>::type
    get(const Matrix &matrix, size_t row) {
        return matrix.row_begin(row);
    }
};

//---------------------------------------------------------------------------
// Builtin backend definition
//---------------------------------------------------------------------------
template <typename real>
struct builtin {
    typedef real value_type;
    typedef long index_type;

    typedef crs<value_type, index_type> matrix;
    typedef std::vector<value_type>     vector;

    struct params {};

    static boost::shared_ptr<matrix>
    copy_matrix(boost::shared_ptr<matrix> A, const params&)
    {
        return A;
    }

    static boost::shared_ptr<vector>
    copy_vector(boost::shared_ptr< vector > x, const params&)
    {
        return x;
    }

    static boost::shared_ptr<vector>
    copy_vector(const vector x, const params&)
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
struct spmv_impl< crs<V, C, P>, std::vector<V> >
{
    typedef crs<V, C, P>   matrix;
    typedef std::vector<V> vector;

    static void apply(V alpha, const matrix &A, const vector &x,
            V beta, vector &y)
    {
        const size_t n = rows(A);
        if (beta) {
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                V sum = 0;
                for(typename matrix::row_iterator a = A.row_begin(i); a; ++a)
                    sum += a.value() * x[ a.col() ];
                y[i] = alpha * sum + beta * y[i];
            }
        } else {
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                V sum = 0;
                for(typename matrix::row_iterator a = A.row_begin(i); a; ++a)
                    sum += a.value() * x[ a.col() ];
                y[i] = alpha * sum;
            }
        }
    }
};

template < typename V, typename C, typename P >
struct residual_impl< crs<V, C, P>, std::vector<V> >
{
    typedef crs<V, C, P>   matrix;
    typedef std::vector<V> vector;

    static void apply(const vector &rhs, const matrix &A, const vector &x,
            vector &r)
    {
        const size_t n = rows(A);
#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            V sum = 0;
            for(typename matrix::row_iterator a = A.row_begin(i); a; ++a)
                sum += a.value() * x[ a.col() ];
            r[i] = rhs[i] - sum;
        }
    }
};

template < typename V >
struct clear_impl< std::vector<V> >
{
    static void apply(std::vector<V> &x)
    {
        const size_t n = x.size();
#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            x[i] = 0;
        }
    }
};

template < typename V >
struct inner_product_impl< std::vector<V> >
{
    static V get(const std::vector<V> &x, const std::vector<V> &y)
    {
        const size_t n = x.size();
        V sum = 0;

#pragma omp parallel for reduction(+:sum)
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            sum += x[i] * y[i];
        }

        return sum;
    }
};

template < typename V >
struct axpby_impl< std::vector<V> > {
    static void apply(V a, const std::vector<V> &x, V b, std::vector<V> &y)
    {
        const size_t n = x.size();
        if (b) {
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                y[i] = a * x[i] + b * y[i];
            }
        } else {
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                y[i] = a * x[i];
            }
        }
    }
};

template < typename V >
struct vmul_impl< std::vector<V> > {
    static void apply(V a, const std::vector<V> &x, const std::vector<V> &y,
            V b, std::vector<V> &z)
    {
        const size_t n = x.size();
        if (b) {
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                z[i] = a * x[i] * y[i] + b * z[i];
            }
        } else {
#pragma omp parallel for
            for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
                z[i] = a * x[i] * y[i];
            }
        }
    }
};

template < typename V >
struct copy_impl< std::vector<V> > {
    static void apply(const std::vector<V> &x, std::vector<V> &y)
    {
        const size_t n = x.size();
#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            y[i] = x[i];
        }
    }
};

} // namespace backend
} // namespace amgcl

#endif
