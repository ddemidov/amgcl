#ifndef AMGCL_DETAIL_QR_HPP
#define AMGCL_DETAIL_QR_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/detail/qr.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  QR decomposition.
 *
 * This is a port of ZGEQR2 procedure from LAPACK and its dependencies.
 * The original code included the following copyright notice.
 * \verbatim
   Copyright (c) 1992-2013 The University of Tennessee and The University
                           of Tennessee Research Foundation.  All rights
                           reserved.
   Copyright (c) 2000-2013 The University of California Berkeley. All
                           rights reserved.
   Copyright (c) 2006-2013 The University of Colorado Denver.  All rights
                           reserved.

   $COPYRIGHT$

   Additional copyrights may follow

   $HEADER$

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   - Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer listed
     in this license in the documentation and/or other materials
     provided with the distribution.

   - Neither the name of the copyright holders nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

   The copyright holders provide no reassurances that the source code
   provided does not infringe any patent, copyright, or any other
   intellectual property rights of third parties.  The copyright holders
   disclaim any liability to any recipient for claims brought against
   recipient by any third party for infringement of that parties
   intellectual property rights.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endverbatim
 */

#include <vector>
#include <cmath>

#include <amgcl/util.hpp>
#include <amgcl/value_type/interface.hpp>

namespace amgcl {
namespace detail {

template <class T, class Enable = void> struct dense_vector_ref;
template <class T, class Enable = void> struct dense_matrix_ref;

template <class T>
struct dense_vector_ref<T, typename boost::disable_if< math::is_static_matrix<T> >::type >
{
    dense_vector_ref(T *ptr, int size, int stride = 1)
        : ptr(ptr), n(size), stride(stride) {}

    int size() const { return n; }

    T& operator[](int i) {
        return ptr[i * stride];
    }

    T operator[](int i) const {
        return ptr[i * stride];
    }

    private:
        T *ptr;
        int n, stride;
};

template <class T>
struct dense_matrix_ref<T, typename boost::disable_if< math::is_static_matrix<T> >::type >
{
    dense_matrix_ref()
        : ptr(0), n(0), m(0), stride(0) {}

    dense_matrix_ref(T *ptr, int rows, int cols, int stride)
        : ptr(ptr), n(rows), m(cols), stride(stride) {}

    dense_matrix_ref(T *ptr, int rows, int cols)
        : ptr(ptr), n(rows), m(cols), stride(cols) {}

    int rows() const { return n; }
    int cols() const { return m; }

    T& operator()(int i, int j) {
        return ptr[i * stride + j];
    }

    T operator()(int i, int j) const {
        return ptr[i * stride + j];
    }

    dense_matrix_ref submatrix(int i, int j, int rows = -1, int cols = -1) const {
        if (rows < 0) rows = n - i;
        if (cols < 0) cols = m - j;

        return dense_matrix_ref(ptr + i * stride + j, rows, cols, stride);
    }

    dense_vector_ref<T> column(int i, int j, int rows = -1) const {
        if (rows < 0) rows = n - i;
        return dense_vector_ref<T>(ptr + i * stride + j, rows, stride);
    }

    private:
        T *ptr;
        int n, m, stride;
};

/// In-place QR factorization.
template <typename value_type>
class QR {
    public:
        QR() : m(0), n(0) {}

        void compute(unsigned rows, unsigned cols, value_type *a, bool needQ = true)
        {
            /*
             *  Ported from ZGEQR2
             *  ==================
             *
             *  Computes a QR factorization of an matrix A:
             *  A = Q * R.
             *
             *  Arguments
             *  =========
             *
             *  rows    The number of rows of the matrix A.
             *  cols    The number of columns of the matrix A.
             *
             *  A       On entry, the rows by cols matrix A.
             *          On exit, the elements on and above the diagonal of the
             *          array contain the min(m,n) by n upper trapezoidal
             *          matrix R (R is upper triangular if m >= n); the
             *          elements below the diagonal, with the array TAU,
             *          represent the unitary matrix Q as a product of
             *          elementary reflectors (see Further Details).
             *
             *  Further Details
             *  ===============
             *
             *  The matrix Q is represented as a product of elementary reflectors
             *
             *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
             *
             *  Each H(i) has the form
             *
             *     H(i) = I - tau * v * v'
             *
             *  where tau is a value_type scalar, and v is a value_type vector
             *  with v[0:i) = 0 and v[i] = 1; v[i:m) is stored on exit in
             *  A[i+1:m)[i], and tau in tau[i].
             *  ==============================================================
             */
            m = rows;
            n = cols;
            k = std::min(m, n);

            r = a;

            dense_matrix_ref<value_type> A(a, m, n);

            tau.resize(k);

            for(unsigned i = 0; i < k; ++i) {
                // Generate elementary reflector H(i) to annihilate A[i+1:m)[i]
                tau[i] = gen_reflector(m-i, A(i,i), A.column(i+1, i));

                if (i+1 < n) {
                    // Apply H(i)' to A[i:m)[i+1:n) from the left
                    apply_reflector(A.column(i,i), math::adjoint(tau[i]), A.submatrix(i,i+1));
                }
            }

            if (needQ) generate_q();
        }

        // Returns element of the matrix R.
        value_type R(unsigned i, unsigned j) const {
            if (j < i) return math::zero<value_type>();
            return r[i*n + j];
        }

        // Returns element of the matrix Q.
        value_type Q(unsigned i, unsigned j) const {
            return q[i*n + j];
        }

        // Solves the system Q R x = f
        void solve(value_type *f, value_type *x) const {
            dense_matrix_ref<value_type> R(r, m, n);
            dense_matrix_ref<value_type> F(f, m, 1);

            for(unsigned i = 0; i < n; ++i)
                apply_reflector(R.column(i,i), math::adjoint(tau[i]), F.submatrix(i,0));

            std::copy(f, f+n, x);

            for(unsigned i = n; i --> 0;) {
                value_type rii = R(i,i);
                if (math::is_zero(rii)) continue;
                x[i] = math::inverse(rii) * x[i];

                for(unsigned j = 0; j < i; ++j)
                    x[j] -= R(j,i) * x[i];
            }
        }
    private:
        typedef typename math::scalar_of<value_type>::type scalar_type;

        static scalar_type sqr(scalar_type x) { return x * x; }

        unsigned m, n, k;

        value_type *r;
        std::vector<value_type> tau;
        std::vector<value_type> q;

        template <class Vector>
        static value_type gen_reflector(int order, value_type &alpha, Vector &&x) {
            /*
             *  Ported from ZLARFG
             *  ==================
             *
             *  Generates a value_type elementary reflector H of order n, such
             *  that
             *
             *        H' * ( alpha ) = ( beta ),   H' * H = I.
             *             (   x   )   (   0  )
             *
             *  where alpha and beta are scalars, with beta real, and x is an
             *  (n-1)-element value_type vector. H is represented in the form
             *
             *        H = I - tau * ( 1 ) * ( 1 v' ) ,
             *                      ( v )
             *
             *  where tau is a value_type scalar and v is a value_type
             *  (n-1)-element vector. Note that H is not hermitian.
             *
             *  If the elements of x are all zero and alpha is real,
             *  then tau = 0 and H is taken to be the unit matrix.
             *
             *  Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
             *
             *  Arguments
             *  =========
             *
             *  order   The order of the elementary reflector.
             *
             *  alpha   On entry, the value alpha.
             *          On exit, it is overwritten with the value beta.
             *
             *  x       dimension (1+(order-2)*abs(stride))
             *          On entry, the vector x.
             *          On exit, it is overwritten with the vector v.
             *
             *  stride  The increment between elements of x.
             *
             *  Returns the value tau.
             *
             *  ==============================================================
             */
            value_type tau = math::zero<value_type>();
            if (order <= 1) return tau;
            int n = order - 1;

            scalar_type xnorm2 = 0;
            for(int i = 0; i < n; ++i)
                xnorm2 += sqr(math::norm(x[i]));

            if (math::is_zero(xnorm2)) return tau;

            scalar_type beta = sqrt(sqr(math::norm(alpha)) + xnorm2);

            tau = math::identity<value_type>() - math::inverse(beta) * alpha;
            alpha = math::inverse(alpha - beta * math::identity<value_type>());

            for(int i = 0; i < n; ++i)
                x[i] = alpha * x[i];

            alpha = beta * math::identity<value_type>();
            return tau;
        }

        template <class Vector, class Matrix>
        static void apply_reflector(const Vector &v, value_type tau, Matrix &&C)
        {
            /*
             *  Ported from ZLARF
             *  =================
             *
             *  Applies an elementary reflector H to an m-by-n matrix C from
             *  the left. H is represented in the form
             *
             *        H = I - v * tau * v'
             *
             *  where tau is a value_type scalar and v is a value_type vector.
             *
             *  If tau = 0, then H is taken to be the unit matrix.
             *
             *  To apply H' (the conjugate transpose of H), supply adjoint(tau)
             *  instead of tau.
             *
             *  Arguments
             *  =========
             *
             *  v        The vector v in the representation of H.
             *           v is not used if tau = 0.
             *           The value of v[0] is ignored and assumed to be 1.
             *
             *  tau      The value tau in the representation of H.
             *
             *  C        On entry, the m-by-n matrix C.
             *           On exit, C is overwritten by the matrix H * C.
             *
             *  c_stride The increment between the rows of C.
             *
             *  ==============================================================
             */

            if (math::is_zero(tau)) return;

            int m = C.rows();
            int n = C.cols();

            // w = C` * v; C -= tau * v * w`
            for(int i = 0; i < n; ++i) {
                value_type s = math::adjoint(C(0,i));
                for(int j = 1; j < m; ++j)
                    s += math::adjoint(C(j,i)) * v[j];

                s = tau * math::adjoint(s);

                C(0,i) -= s;
                for(int j = 1; j < m; ++j)
                    C(j,i) -= v[j] * s;
            }
        }

        void generate_q() {
            /*
             *  Ported from ZUNG2R
             *  ==================
             *
             *  Generates an m by n matrix Q with orthonormal columns, which is
             *  defined as the first n columns of a product of k elementary
             *  reflectors of order m
             *
             *        Q  =  H(1) H(2) . . . H(k)
             *
             *  as returned by compute() [ZGEQR2].
             *
             *  ==============================================================
             */
            q.resize(m * n);
            dense_matrix_ref<value_type> Q(q.data(), m, n);
            dense_matrix_ref<value_type> R(r, m, n);

            // Initialise columns k+1:n to zero.
            // [In the original code these were initialized to the columns of
            // the unit matrix, but since k = min(n,m), the main diagonal is
            // never seen here].
            for(unsigned i = 0; i < m; ++i)
                for(unsigned j = k; j < n; ++j)
                    Q(i,j) = math::zero<value_type>();

            for(unsigned i = k; i --> 0; ) {
                // Apply H(i) to A[i:m)[i+1:n) from the left
                if (i+1 < n)
                    apply_reflector(R.column(i,i), tau[i], Q.submatrix(i,i+1));

                // Copy i-th reflector (including zeros and unit diagonal)
                // to the column of Q to be processed next
                for(unsigned j = 0; j < i; ++j)
                    Q(j,i) = math::zero<value_type>();

                Q(i,i) = math::identity<value_type>() - tau[i];

                for(unsigned j = i + 1; j < m; ++j)
                    Q(j,i) = -tau[i] * R(j,i);
            }
        }
};

} // namespace detail
} // namespace amgcl

#endif
