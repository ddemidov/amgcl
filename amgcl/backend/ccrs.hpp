#ifndef AMGCL_BACKEND_CCRS_HPP
#define AMGCL_BACKEND_CCRS_HPP

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
 * \file   amgcl/backend/ccrs.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Sparse matrix in CRS format with lossy compression.
 *
 * Each row is stored as a pointer to a table of unique matrix rows, where
 * uniqueness is determined approximately.  This may result in significant
 * storage space savings in case matrix has regular structure (e.g. it
 * represents a Poisson problem in a domain with piecewise-constant or slowly
 * changing properties).  The precision loss is possible, but may not be
 * important (e.g.  coefficients come from an experiment with incorporated
 * observation error).
 */

#include <vector>
#include <deque>
#include <memory>
#include <numeric>

#include <boost/static_assert.hpp>
#include <boost/functional/hash.hpp>
#include <boost/range.hpp>
#include <boost/range/numeric.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

#include <amgcl/util.hpp>
#include <amgcl/backend/interface.hpp>

namespace amgcl {
namespace backend {

namespace detail {

template <typename To, typename From>
To reinterpret(From from) {
    BOOST_STATIC_ASSERT_MSG(
            sizeof(To) == sizeof(From),
            "Incompatible types in reinterpret()"
            );
    return *reinterpret_cast<const To*>(&from);
}

template <typename T, class Enable = void>
struct hash_impl {
    static inline size_t get(T v) {
        return boost::hash<T>()(v);
    }
};

template <>
struct hash_impl<double> {
    // Zero-out least significant bits and hash as uint64_t:
    static inline size_t get(double v) {
        static const uint64_t mask = 0xffffffffff000000;
                                    // eeefffffffffffff;
        return boost::hash<uint64_t>()(reinterpret<uint64_t>(v) & mask);
    }
};

template <>
struct hash_impl<float> {
    // Zero-out least significant bits and hash as uint64_t:
    static inline size_t get(float v) {
        static const uint32_t mask = 0xfffffc00;
                                    // eeffffff;
        return boost::hash<uint32_t>()(reinterpret<uint32_t>(v) & mask);
    }
};

template <class T>
inline size_t hash(const T &v) {
    return hash_impl<T>::get(v);
}

} // namespace detail

/// Lossy compression for CSR sparse matrix format.
/**
 * Each row is stored as a pointer to a table of unique matrix rows, where
 * uniqueness is determined approximately.  This may result in significant
 * storage space savings in case matrix has regular structure (e.g. it
 * represents a Poisson problem in a domain with piecewise-constant or slowly
 * changing properties).  The precision loss is possible, but may not be
 * important (e.g.  coefficients come from an experiment with incorporated
 * observation error).
*/
template <
    typename val_t = double,
    typename col_t = int,
    typename ptr_t = col_t
    >
class ccrs {
    BOOST_STATIC_ASSERT_MSG(
            boost::is_signed<col_t>::value,
            "Column type should be signed!"
            );

    private:
        size_t nrows, ncols, nnz;
        val_t  eps;

        std::vector< ptr_t > idx;
        std::vector< ptr_t > ptr;
        std::vector< col_t > col;
        std::vector< val_t > val;

        // Returns operand incremented by a given value.
        struct shift {
            typedef col_t result_type;

            col_t s;

            shift(col_t s) : s(s) {}

            col_t operator()(col_t v) const {
                return v + s;
            }
        };

        // Extracts and stores unique rows.
        struct builder_t {
            std::deque< ptr_t > idx;
            std::deque< ptr_t > ptr;
            std::deque< col_t > col;
            std::deque< val_t > val;

            // Hashes and compares matrix rows.
            struct row_hasher {
                val_t eps;

                row_hasher(val_t eps) : eps(eps) {}

                struct do_hash {
                    template <class Tuple>
                    size_t operator()(size_t h, Tuple t) const {
                        return h
                            ^ detail::hash(boost::get<0>(t))
                            ^ detail::hash(boost::get<1>(t));
                    }
                };

                struct cmp {
                    val_t eps;

                    cmp(val_t eps) : eps(eps) {}

                    template <class Tuple1, class Tuple2>
                    bool operator()(Tuple1 x, Tuple2 y) const {
                        return
                            (boost::get<0>(x) == boost::get<0>(y)) &&
                            (std::fabs(boost::get<1>(x) - boost::get<1>(y)) < eps);
                    }
                };

                template <class Range>
                size_t operator()(const Range &r) const {
                    return boost::accumulate(r, boost::size(r), do_hash());
                }

                template <class Range1, class Range2>
                bool operator()(const Range1 &r1, const Range2 &r2) const {
                    return boost::equal(r1, r2, cmp(eps));
                }
            } hasher;

            typedef boost::zip_iterator<
                        boost::tuple<
                            typename std::deque< col_t >::const_iterator,
                            typename std::deque< val_t >::const_iterator
                        >
                    > row_iterator;

            typedef boost::iterator_range< row_iterator > row_range;

            typedef
                boost::unordered_map<
                    row_range, ptr_t,
                    row_hasher, row_hasher
                    >
                index_type;

            index_type index;

            builder_t(size_t nrows, val_t eps)
                : idx(nrows, 0), hasher(eps), index(1979, hasher, hasher)
            {
                // Artificial empty row:
                ptr.push_back(0);
                ptr.push_back(0);

                index.insert(
                        std::make_pair(
                            boost::make_iterator_range(
                                boost::make_zip_iterator(
                                    boost::make_tuple(col.begin(), val.begin())
                                    ),
                                boost::make_zip_iterator(
                                    boost::make_tuple(col.end(), val.end())
                                    )
                                ),
                            index.size()
                            )
                        );
            }

            void insert(col_t row_begin, col_t row_end,
                    const ptr_t *r, const col_t *c, const val_t *v)
            {
                for(col_t i = row_begin, j = 0; i < row_end; ++i, ++j) {
                    shift s(-i);

                    typename index_type::iterator pos = index.find(
                            boost::make_iterator_range(
                                boost::make_zip_iterator(
                                    boost::make_tuple(
                                        boost::make_transform_iterator(c + r[j], s),
                                        v + r[j])
                                    ),
                                boost::make_zip_iterator(
                                    boost::make_tuple(
                                        boost::make_transform_iterator(c + r[j+1], s),
                                        v + r[j+1])
                                    )
                                ),
                            hasher, hasher);

                    if (pos == index.end()) {
                        idx[i] = index.size();

                        size_t start = val.size();

                        for(size_t k = r[j]; k < r[j+1]; ++k) {
                            col.push_back( c[k] - i );
                            val.push_back( v[k] );
                        }

                        index.insert(
                                std::make_pair(
                                    boost::make_iterator_range(
                                        boost::make_zip_iterator(
                                            boost::make_tuple(
                                                col.begin() + start,
                                                val.begin() + start)
                                            ),
                                        boost::make_zip_iterator(
                                            boost::make_tuple(
                                                col.end(),
                                                val.end()
                                                )
                                            )
                                        ),
                                    idx[i]
                                    )
                                );

                        ptr.push_back(val.size());
                    } else {
                        idx[i] = pos->second;
                    }
                }
            }
        };

        boost::shared_ptr<builder_t> builder;

    public:
        class row_iterator {
            public:
                operator bool() const {
                    return m_col < m_end;
                }

                row_iterator& operator++() {
                    ++m_col;
                    ++m_val;
                }

                col_t col() const {
                    return m_row + *m_col;
                }

                val_t value() const {
                    return *m_val;
                }

            private:
                friend class ccrs;

                col_t         m_row;
                col_t const * m_col;
                col_t const * m_end;
                val_t const * m_val;

                row_iterator(
                        col_t         row,
                        col_t const * col,
                        col_t const * end,
                        val_t const * val
                        ) : m_row(row), m_col(col), m_end(end), m_val(val)
                {}
        };

        /// Constructor.
        ccrs(size_t nrows, size_t ncols, val_t eps = 1e-6)
            : nrows(nrows), ncols(ncols), nnz(0), eps(eps),
              builder(new builder_t(nrows, eps))
        { }

        /// Store matrix slice.
        /**
         * May accept whole matrix, or just a slice of matrix rows.
         */
        void insert(col_t row_begin, col_t row_end,
                const ptr_t *r, const col_t *c, const val_t *v)
        {
            precondition(builder, "Matrix has been finalized");
            builder->insert(row_begin, row_end, r, c, v);
        }

        /// All rows have been processed; finalize the construction phase.
        void finalize() {
            precondition(builder, "Matrix has been finalized");

            idx.assign(builder->idx.begin(), builder->idx.end());
            ptr.assign(builder->ptr.begin(), builder->ptr.end());
            col.assign(builder->col.begin(), builder->col.end());
            val.assign(builder->val.begin(), builder->val.end());

            builder.reset();

            for(size_t i = 0; i < nrows; ++i)
                nnz += ptr[idx[i] + 1] - ptr[idx[i]];
        }

        /// Number of unique rows in the matrix.
        size_t unique_rows() const {
            return ptr.size() - 1;
        }

        /// Returns boost::zip_iterator to start of columns/values range for a given row.
        row_iterator row_begin(size_t i) const {
            assert(!builder && i < nrows);

            ptr_t row = idx[i];
            ptr_t beg = ptr[row];
            ptr_t end = ptr[row+1];

            return row_iterator(i, &col[beg], &col[end], &val[beg]);
        }

        /// Number of rows.
        size_t rows() const {
            return nrows;
        }

        /// Number of cols.
        size_t cols() const {
            return ncols;
        }

        /// Number of nonzeros in the matrix.
        size_t non_zeros() const {
            precondition(!builder, "Matrix has not been finalized");

            return nnz;
        }

        /// Compression ratio.
        double compression() const {
            precondition(!builder, "Matrix has not been finalized");

            return 1.0 *
                (
                 sizeof(idx[0]) * idx.size() +
                 sizeof(ptr[0]) * ptr.size() +
                 sizeof(col[0]) * col.size() +
                 sizeof(val[0]) * val.size()
                ) /
                (
                    sizeof(ptr[0]) * (nrows + 1) +
                    sizeof(col[0]) * nnz        +
                    sizeof(val[0]) * nnz
                );
        }


        /// Matrix transpose.
        friend ccrs transp(const ccrs &A) {
            const col_t chunk_size = 4096;

            col_t lw = 0, rw = 0;
            for(size_t i = 0, n = A.unique_rows(); i < n; ++i) {
                for(ptr_t j = A.ptr[i]; j < A.ptr[i + 1]; ++j) {
                    lw = std::max(lw, -A.col[j]);
                    rw = std::max(rw,  A.col[j]);
                }
            }

            ccrs T(A.ncols, A.nrows, A.eps);

            std::vector<ptr_t> ptr;
            std::vector<col_t> col;
            std::vector<val_t> val;

            for(col_t chunk = 0; chunk < A.nrows; chunk += chunk_size) {
                col_t row_start = std::max(chunk - rw,              static_cast<col_t>(0));
                col_t row_end   = std::min(chunk + chunk_size + lw, static_cast<col_t>(A.nrows));
                col_t chunk_end = std::min(chunk + chunk_size,      static_cast<col_t>(A.nrows));

                ptr.clear();
                col.clear();
                val.clear();

                ptr.resize(chunk_size + 1, 0);

                for(col_t i = row_start; i < row_end; ++i)
                    for(ccrs::row_iterator j = A.row_begin(i); j; ++j) {
                        col_t c = j.col();
                        if (c >= chunk && c < chunk_end)
                            ++( ptr[c - chunk + 1] );
                    }

                std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

                col.resize(ptr.back());
                val.resize(ptr.back());

                for(size_t i = row_start; i < row_end; ++i) {
                    for(ccrs::row_iterator j = A.row_begin(i); j; ++j) {
                        col_t c = j.col();
                        val_t v = j.value();

                        if (c >= chunk && c < chunk_end) {
                            ptr_t head = ptr[c - chunk]++;

                            col[head] = i;
                            val[head] = v;
                        }
                    }
                }

                std::rotate(ptr.begin(), ptr.end() - 1, ptr.end());
                ptr[0] = 0;

                T.insert(chunk, chunk_end, ptr.data(), col.data(), val.data());
            }

            T.finalize();

            return T;
        }

        static void insertion_sort(col_t *col, val_t *val, int n) {
            for(int j = 1; j < n; ++j) {
                col_t c = col[j];
                val_t v = val[j];
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

        /// Matrix-matrix product.
        friend ccrs prod(const ccrs &A, const ccrs &B) {
            ccrs C(A.nrows, B.ncols, std::min(A.eps, B.eps));

            std::vector<col_t> marker(B.ncols, -1);

            ptr_t ptr[2] = {0, 0};
            std::vector<col_t> col;
            std::vector<val_t> val;

            for(size_t ia = 0; ia < A.nrows; ++ia) {
                col.clear();
                val.clear();

                for(ccrs::row_iterator ja = A.row_begin(ia); ja; ++ja) {
                    col_t ca = ja.col();
                    val_t va = ja.value();

                    for(ccrs::row_iterator jb = B.row_begin(ca); jb; ++jb) {
                        col_t cb = jb.col();
                        val_t vb = jb.value();

                        if (marker[cb] < 0) {
                            marker[cb] = col.size();
                            col.push_back(cb);
                            val.push_back(va * vb);
                        } else {
                            val[marker[cb]] += va * vb;
                        }
                    }
                }

                for(typename std::vector<col_t>::iterator c = col.begin(); c != col.end(); ++c)
                    marker[*c] = -1;

                ptr[1] = col.size();

                insertion_sort(col.data(), val.data(), col.size());

                C.insert(ia, ia + 1, ptr, col.data(), val.data());
            }

            C.finalize();
            return C;
        }

};

//---------------------------------------------------------------------------
// Specialization of matrix interface
//---------------------------------------------------------------------------
template < typename V, typename C, typename P >
struct value_type< ccrs<V, C, P> > {
    typedef V type;
};

template < typename V, typename C, typename P >
struct rows_impl< ccrs<V, C, P> > {
    static size_t get(const ccrs<V, C, P> &A) {
        return A.rows();
    }
};

template < typename V, typename C, typename P >
struct cols_impl< ccrs<V, C, P> > {
    static size_t get(const ccrs<V, C, P> &A) {
        return A.cols();
    }
};

template < typename V, typename C, typename P >
struct row_iterator< ccrs<V, C, P> > {
    typedef
        typename ccrs<V, C, P>::row_iterator
        type;
};

template < typename V, typename C, typename P >
struct row_begin_impl< ccrs<V, C, P> > {
    typedef ccrs<V, C, P> Matrix;
    static typename backend::row_iterator<Matrix>::type
    get(const Matrix &matrix, size_t row) {
        return matrix.row_begin(row);
    }
};

//---------------------------------------------------------------------------
// compressed_csr backend definition
//---------------------------------------------------------------------------
template <typename real>
struct compressed_crs {
    typedef real value_type;
    typedef long index_type;

    typedef ccrs<real, index_type, index_type> matrix;
    typedef typename builtin<real>::vector     vector;

    struct params {
        real eps;

        params(real eps = 1e-6) : eps(eps) {}
    };

    static boost::shared_ptr<matrix>
    copy_matrix(boost::shared_ptr< typename backend::builtin<real>::matrix > A,
            const params &prm)
    {
        const size_t n = rows(*A);
        const size_t m = cols(*A);

        boost::shared_ptr<matrix> C = boost::make_shared<matrix>(n, m, prm.eps);
        C->insert(0, n, A->ptr.data(), A->col.data(), A->val.data());
        C->finalize();

        return C;
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
struct spmv_impl< ccrs<V, C, P>, std::vector<V> >
{
    typedef ccrs<V, C, P>  matrix;
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
struct residual_impl< ccrs<V, C, P>, std::vector<V> >
{
    typedef ccrs<V, C, P>  matrix;
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

} // namespace backend
} // namespace amgcl


#endif
