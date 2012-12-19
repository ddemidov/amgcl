#ifndef AMGCL_LEVEL_CUDA_HPP
#define AMGCL_LEVEL_CUDA_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   level_cuda.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Level of an AMG hierarchy for use with Thrust/CUSPARSE combination.
 * Only supports standalone solution for now.
 */

#include <stdexcept>

#include <boost/static_assert.hpp>
#include <boost/array.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/type_traits/is_same.hpp>

#include <amgcl/level_params.hpp>
#include <amgcl/spmat.hpp>
#include <amgcl/common.hpp>

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <cusparse_v2.h>

namespace amgcl {

namespace sparse {

/// Possible matrix storage formats.
enum cuda_matrix_format {
    CUDA_MATRIX_CRS, ///< Compressed matrix. Fastest construction, slowest operation (on a GPU). Best suited for CPU compute devices.
    CUDA_MATRIX_HYB  ///< Hybrid ELL format. Best choice for general matrices on GPU compute devices.
};

/// Wrapper around CUSPARSE matrix
template <typename value_type, cuda_matrix_format>
class cuda_matrix;

/// Wrapper around CUSPARSE matrix in Hybrid format
template <typename value_type>
class cuda_matrix<value_type, CUDA_MATRIX_HYB> {
    public:
        /// Empty constructor.
        cuda_matrix() : rows(0), cols(0), nnz(0), desc(0), mat(0) {}

        /// Convert sparse matrix to cuda_matrix.
        template <class spmat>
        cuda_matrix(const spmat &A)
            : rows(matrix_rows(A)),
              cols(matrix_cols(A)),
              nnz (matrix_nonzeros(A)),
              desc(0), mat(0)
        {
            typedef typename matrix_index<spmat>::type index_t;
            typedef typename matrix_value<spmat>::type value_t;

            BOOST_STATIC_ASSERT_MSG((boost::is_same<value_type, value_t>::value),
                    "Matrix has wrong value type");

            cusparseCreate();

            check(cusparseCreateMatDescr(&desc),
                    "cusparseCreateMatDescr failed");

            check(cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL),
                    "cusparseSetMatType failed");

            check(cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO),
                    "cusparseSetMatIndexBase failed");

            check(cusparseCreateHybMat(&mat), "cusparseCreateHybMat failed");

            BOOST_AUTO(Arow, sparse::matrix_outer_index(A));
            BOOST_AUTO(Acol, sparse::matrix_inner_index(A));
            BOOST_AUTO(Aval, sparse::matrix_values(A));

            convert_to_hybrid(
                    thrust::device_vector<int>    (Arow, Arow + rows + 1),
                    thrust::device_vector<int>    (Acol, Acol + nnz),
                    thrust::device_vector<value_t>(Aval, Aval + nnz)
                    );
        }

        ~cuda_matrix() {
            if (mat) check(cusparseDestroyHybMat(mat),
                    "cusparseDestroyHybMat failed");

            if (desc) check(cusparseDestroyMatDescr(desc),
                    "cusparseDestroyMatDescr failed");
        }

        /// Matrix-vector product.
        /**
         * \f[y = \alpha A x + \beta y \f]
         */
        void mul(float alpha, const thrust::device_vector<float> &x,
                float beta, thrust::device_vector<float> &y) const
        {
            BOOST_STATIC_ASSERT_MSG((boost::is_same<float, value_type>::value),
                    "Wrong vector type in matrix-vector product");

            check(cusparseShybmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, desc, mat,
                        thrust::raw_pointer_cast(&x[0]), &beta,
                        thrust::raw_pointer_cast(&y[0])
                        ), "cusparseShybmv failed"
                 );
        }

        /// Matrix-vector product.
        /**
         * \f[y = \alpha A x + \beta y \f]
         */
        void mul(double alpha, const thrust::device_vector<double> &x,
                double beta, thrust::device_vector<double> &y) const
        {
            BOOST_STATIC_ASSERT_MSG((boost::is_same<double, value_type>::value),
                    "Wrong vector type in matrix-vector product");

            check(cusparseDhybmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, desc, mat,
                        thrust::raw_pointer_cast(&x[0]), &beta,
                        thrust::raw_pointer_cast(&y[0])
                        ), "cusparseDhybmv failed"
                 );
        }

        /// Number of rows in the matrix.
        unsigned size() const {
            return rows;
        }

        /// Number of nonzero entries in the matrix.
        unsigned nonzeros() const {
            return nnz;
        }

        /// Initialization of CUSPARSE context.
        /**
         * This is called automatically by cuda_matrix constructor.
         */
        static void cusparseCreate() {
            if (!cusp_handle)
                check(::cusparseCreate(&cusp_handle), "cusparseCreate failed");
        }

        /// Releases of resources used by CUSPARSE library.
        /**
         * There is probably no need to call this.
         */
        static void cusparseDestroy() {
            if (cusp_handle)
                check(::cusparseDestroy(cusp_handle), "cusparseDestroy failed");
        }
    private:
        unsigned rows, cols, nnz;
        cusparseMatDescr_t desc;
        cusparseHybMat_t   mat;

        static cusparseHandle_t cusp_handle;

        inline static void check(cusparseStatus_t status, const char *err_msg) {
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(err_msg);
        }

        void convert_to_hybrid(
                const thrust::device_vector<int>   &row,
                const thrust::device_vector<int>   &col,
                const thrust::device_vector<float> &val
                )
        {
            check(cusparseScsr2hyb(cusp_handle, rows, cols, desc,
                        thrust::raw_pointer_cast(&val[0]),
                        thrust::raw_pointer_cast(&row[0]),
                        thrust::raw_pointer_cast(&col[0]),
                        mat, 0, CUSPARSE_HYB_PARTITION_AUTO
                        ),
                    "cusparseScsr2hyb failed"
                 );
        }

        void convert_to_hybrid(
                const thrust::device_vector<int>    &row,
                const thrust::device_vector<int>    &col,
                const thrust::device_vector<double> &val
                )
        {
            check(cusparseDcsr2hyb(cusp_handle, rows, cols, desc,
                        thrust::raw_pointer_cast(&val[0]),
                        thrust::raw_pointer_cast(&row[0]),
                        thrust::raw_pointer_cast(&col[0]),
                        mat, 0, CUSPARSE_HYB_PARTITION_AUTO
                        ),
                    "cusparseDcsr2hyb failed"
                 );
        }
};

/// Wrapper around CUSPARSE matrix in CRS format
template <typename value_type>
class cuda_matrix<value_type, CUDA_MATRIX_CRS> {
    public:
        /// Empty constructor.
        cuda_matrix() : rows(0), cols(0), nnz(0), desc(0) {}

        /// Convert sparse matrix to cuda_matrix.
        template <class spmat>
        cuda_matrix(const spmat &A)
            : rows(matrix_rows(A)),
              cols(matrix_cols(A)),
              nnz (matrix_nonzeros(A)),
              row(matrix_outer_index(A), matrix_outer_index(A) + rows + 1),
              col(matrix_inner_index(A), matrix_inner_index(A) + nnz),
              val(matrix_values(A), matrix_values(A) + nnz),
              desc(0)
        {
            typedef typename matrix_index<spmat>::type index_t;
            typedef typename matrix_value<spmat>::type value_t;

            cusparseCreate();

            check(cusparseCreateMatDescr(&desc),
                    "cusparseCreateMatDescr failed");

            check(cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL),
                    "cusparseSetMatType failed");

            check(cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO),
                    "cusparseSetMatIndexBase failed");
        }

        ~cuda_matrix() {
            if (desc) check(cusparseDestroyMatDescr(desc),
                    "cusparseDestroyMatDescr failed");
        }

        /// Matrix-vector product.
        /**
         * \f[y = \alpha A x + \beta y \f]
         */
        void mul(float alpha, const thrust::device_vector<float> &x,
                float beta, thrust::device_vector<float> &y) const
        {
            BOOST_STATIC_ASSERT_MSG((boost::is_same<float, value_type>::value),
                    "Wrong vector type in matrix-vector product");

            check(cusparseScsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        rows, cols, nnz, &alpha, desc,
                        thrust::raw_pointer_cast(&val[0]),
                        thrust::raw_pointer_cast(&row[0]),
                        thrust::raw_pointer_cast(&col[0]),
                        thrust::raw_pointer_cast(&x[0]), &beta,
                        thrust::raw_pointer_cast(&y[0])
                        ), "cusparseShybmv failed"
                 );
        }

        /// Matrix-vector product.
        /**
         * \f[y = \alpha A x + \beta y \f]
         */
        void mul(double alpha, const thrust::device_vector<double> &x,
                double beta, thrust::device_vector<double> &y) const
        {
            BOOST_STATIC_ASSERT_MSG((boost::is_same<double, value_type>::value),
                    "Wrong vector type in matrix-vector product");

            check(cusparseDcsrmv(cusp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        rows, cols, nnz, &alpha, desc,
                        thrust::raw_pointer_cast(&val[0]),
                        thrust::raw_pointer_cast(&row[0]),
                        thrust::raw_pointer_cast(&col[0]),
                        thrust::raw_pointer_cast(&x[0]), &beta,
                        thrust::raw_pointer_cast(&y[0])
                        ), "cusparseShybmv failed"
                 );
        }

        /// Number of rows in the matrix.
        unsigned size() const {
            return rows;
        }

        /// Number of nonzero entries in the matrix.
        unsigned nonzeros() const {
            return nnz;
        }

        /// Initialization of CUSPARSE context.
        /**
         * This is called automatically by cuda_matrix constructor.
         */
        static void cusparseCreate() {
            if (!cusp_handle)
                check(::cusparseCreate(&cusp_handle), "cusparseCreate failed");
        }

        /// Releases of resources used by CUSPARSE library.
        /**
         * There is probably no need to call this.
         */
        static void cusparseDestroy() {
            if (cusp_handle)
                check(::cusparseDestroy(cusp_handle), "cusparseDestroy failed");
        }
    private:
        unsigned rows, cols, nnz;

        thrust::device_vector<int>        row;
        thrust::device_vector<int>        col;
        thrust::device_vector<value_type> val;

        cusparseMatDescr_t desc;

        static cusparseHandle_t cusp_handle;

        inline static void check(cusparseStatus_t status, const char *err_msg) {
            if (status != CUSPARSE_STATUS_SUCCESS)
                throw std::runtime_error(err_msg);
        }
};


template <typename value_type>
cusparseHandle_t cuda_matrix<value_type, CUDA_MATRIX_HYB>::cusp_handle = 0;

template <typename value_type>
cusparseHandle_t cuda_matrix<value_type, CUDA_MATRIX_CRS>::cusp_handle = 0;

} // namespace sparse

template <typename T>
struct axpy {
    T a;
    axpy(T a) : a(a) {}

    __device__ __host__ T operator()(T x, T y) const {
        return a * x + y;
    }
};

namespace level {

/// Thrust/CUSPARSE based AMG hierarchy.
/**
 * Level of an AMG hierarchy for use with Thrust/CUSPARSE data structures. Uses
 * NVIDIA CUDA technology for acceleration.
 * \ingroup levels
 */
template <sparse::cuda_matrix_format format = sparse::CUDA_MATRIX_HYB>
struct cuda {

/// Parameters for CUSPARSE-based level storage scheme.
class params : public amgcl::level::params {};

template <typename T>
static T norm(const thrust::device_vector<T> &x) {
    return sqrt(thrust::inner_product(
        x.begin(), x.end(), x.begin(), static_cast<T>(0)
        ));
}

template <typename value_t, typename index_t = long long>
class instance {
    public:
        typedef sparse::matrix<value_t, index_t> cpu_matrix;
        typedef sparse::cuda_matrix<value_t, format> matrix;

        instance(cpu_matrix &a, cpu_matrix &p, cpu_matrix &r, const params &prm, unsigned nlevel)
            : A(a), P(p), R(r), d(a.rows), t(a.rows)
        {
            {
                BOOST_AUTO(dia, diagonal(a));
                thrust::copy(dia.begin(), dia.end(), d.begin());
            }

            if (nlevel) {
                u.resize(a.rows);
                f.resize(a.rows);

                if (prm.kcycle && nlevel % prm.kcycle == 0)
                    for(BOOST_AUTO(v, cg.begin()); v != cg.end(); ++v)
                        v->resize(a.rows);
            }

            a.clear();
            p.clear();
            r.clear();
        }

        instance(cpu_matrix &a, cpu_matrix &ai, const params &prm, unsigned nlevel)
            : A(a), Ainv(ai), d(a.rows), u(a.rows), f(a.rows), t(a.rows)
        {
            {
                BOOST_AUTO(dia, diagonal(a));
                thrust::copy(dia.begin(), dia.end(), d.begin());
            }

            a.clear();
            ai.clear();
        }

        // Returns reference to the system matrix
        const matrix& get_matrix() const {
            return A;
        }

        struct relax_functor {
            value_t w;

            relax_functor(value_t w) : w(w) {}

            template <class T>
            __host__ __device__
            void operator()(T zip) const {
                value_t t = thrust::get<1>(zip);
                value_t d = thrust::get<2>(zip);

                thrust::get<0>(zip) += w * t / d;
            }
        };

        // Perform one relaxation (smoothing) step.
        void relax(const thrust::device_vector<value_t> &rhs,
                thrust::device_vector<value_t> &x, const params &prm) const
        {
            const index_t n = x.size();

            thrust::copy(rhs.begin(), rhs.end(), t.begin());;
            A.mul(-1, x, 1, t);

            thrust::for_each(
                    thrust::make_zip_iterator(
                        thrust::make_tuple( x.begin(), t.begin(), d.begin() )
                        ),
                    thrust::make_zip_iterator(
                        thrust::make_tuple( x.end(), t.end(), d.end() )
                        ),
                    relax_functor(prm.relax_factor)
                    );
        }

        // Compute residual value.
        value_t resid(const thrust::device_vector<value_t> &rhs,
                thrust::device_vector<value_t> &x) const
        {
            thrust::copy(rhs.begin(), rhs.end(), t.begin());;
            A.mul(-1, x, 1, t);

            return norm(t);
        }

        // Perform one V-cycle. Coarser levels are cycled recursively. The
        // coarsest level is solved directly.
        template <class Iterator>
        static void cycle(Iterator plvl, Iterator end, const params &prm,
                const thrust::device_vector<value_t> &rhs, thrust::device_vector<value_t> &x)
        {
            Iterator pnxt = plvl; ++pnxt;

            instance *lvl = plvl->get();
            instance *nxt = pnxt->get();

            if (pnxt != end) {
                for(unsigned j = 0; j < prm.ncycle; ++j) {
                    for(unsigned i = 0; i < prm.npre; ++i) lvl->relax(rhs, x, prm);

                    thrust::copy(rhs.begin(), rhs.end(), lvl->t.begin());;
                    lvl->A.mul(-1, x, 1, lvl->t);
                    lvl->R.mul(1, lvl->t, 0, nxt->f);
                    thrust::fill(nxt->u.begin(), nxt->u.end(), static_cast<value_t>(0));

                    if (nxt->cg[0].size())
                        kcycle(pnxt, end, prm, nxt->f, nxt->u);
                    else
                        cycle(pnxt, end, prm, nxt->f, nxt->u);

                    lvl->P.mul(1, nxt->u, 1, x);

                    for(unsigned i = 0; i < prm.npost; ++i) lvl->relax(rhs, x, prm);
                }
            } else {
                lvl->Ainv.mul(1, rhs, 0, x);
            }
        }

        template <class Iterator>
        static void kcycle(Iterator plvl, Iterator end, const params &prm,
                const thrust::device_vector<value_t> &rhs, thrust::device_vector<value_t> &x)
        {
            static const value_t zero = 0;

            Iterator pnxt = plvl; ++pnxt;

            instance *lvl = plvl->get();
            instance *nxt = pnxt->get();

            if (pnxt != end) {
                thrust::device_vector<value_t> &r = lvl->cg[0];
                thrust::device_vector<value_t> &s = lvl->cg[1];
                thrust::device_vector<value_t> &p = lvl->cg[2];
                thrust::device_vector<value_t> &q = lvl->cg[3];

                thrust::copy(rhs.begin(), rhs.end(), r.begin());

                value_t rho1 = 0, rho2 = 0;

                for(int iter = 0; iter < 2; ++iter) {
                    thrust::fill(s.begin(), s.end(), zero);
                    cycle(plvl, end, prm, r, s);

                    rho2 = rho1;
                    rho1 = thrust::inner_product(r.begin(), r.end(), s.begin(), zero);

                    if (iter)
                        thrust::transform(p.begin(), p.end(), s.begin(), p.begin(),
                                axpy<value_t>(rho1 / rho2));
                    else
                        thrust::copy(s.begin(), s.end(), p.begin());

                    lvl->A.mul(1, p, 0, q);

                    value_t alpha = rho1 / thrust::inner_product(q.begin(), q.end(), p.begin(), zero);

                    thrust::transform(p.begin(), p.end(), x.begin(), x.begin(),
                            axpy<value_t>(alpha));
                    thrust::transform(q.begin(), q.end(), r.begin(), r.begin(),
                            axpy<value_t>(-alpha));
                }
            } else {
                lvl->Ainv.mul(1, rhs, 0, x);
            }
        }

        index_t size() const {
            return A.size();
        }

        index_t nonzeros() const {
            return A.nonzeros();
        }
    private:
        sparse::cuda_matrix<value_t, format> A;
        sparse::cuda_matrix<value_t, format> P;
        sparse::cuda_matrix<value_t, format> R;
        sparse::cuda_matrix<value_t, format> Ainv;

        thrust::device_vector<value_t> d;

        mutable boost::array<thrust::device_vector<value_t>, 4> cg;

        mutable thrust::device_vector<value_t> u;
        mutable thrust::device_vector<value_t> f;
        mutable thrust::device_vector<value_t> t;
};

};

} // namespace level


/// Conjugate Gradient method for Thrust/CUSPARSE combination.
/**
 * Implementation is based on \ref Templates_1994 "Barrett (1994)"
 *
 * \param A   The system matrix.
 * \param rhs The right-hand side.
 * \param P   The preconditioner. Should provide apply(rhs, x) method.
 * \param x   The solution. Contains an initial approximation on input, and
 *            the approximated solution on output.
 * \param prm The control parameters.
 *
 * \returns a pair containing number of iterations made and precision
 * achieved.
 *
 * \ingroup iterative
 */
template <class value_t, sparse::cuda_matrix_format format, class precond>
std::pair< int, value_t > solve(
        const sparse::cuda_matrix<value_t, format> &A,
        const thrust::device_vector<value_t> &rhs,
        const precond &P,
        thrust::device_vector<value_t> &x,
        cg_tag prm = cg_tag()
        )
{
    const size_t n = x.size();
    static const value_t zero = 0;

    thrust::device_vector<value_t> r(n), s(n), p(n), q(n);

    thrust::copy(rhs.begin(), rhs.end(), r.begin());
    A.mul(-1, x, 1, r);

    value_t rho1 = 0, rho2 = 0;
    value_t norm_of_rhs = level::cuda<format>::norm(rhs);

    if (norm_of_rhs == 0) {
        thrust::fill(x.begin(), x.end(), zero);
        return std::make_pair(0, norm_of_rhs);
    }

    int     iter;
    value_t res;
    for(
            iter = 0;
            (res = level::cuda<format>::norm(r) / norm_of_rhs) > prm.tol && iter < prm.maxiter;
            ++iter
       )
    {
        thrust::fill(s.begin(), s.end(), zero);
        P.apply(r, s);

        rho2 = rho1;
        rho1 = thrust::inner_product(r.begin(), r.end(), s.begin(), zero);

        if (iter)
            thrust::transform(p.begin(), p.end(), s.begin(), p.begin(),
                    axpy<value_t>(rho1 / rho2));
        else
            thrust::copy(s.begin(), s.end(), p.begin());

        A.mul(1, p, 0, q);

        value_t alpha = rho1 / thrust::inner_product(q.begin(), q.end(), p.begin(), zero);

        thrust::transform(p.begin(), p.end(), x.begin(), x.begin(),
                axpy<value_t>(alpha));
        thrust::transform(q.begin(), q.end(), r.begin(), r.begin(),
                axpy<value_t>(-alpha));
    }

    return std::make_pair(iter, res);
}


} // namespace amgcl

#endif
