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

/// In-place QR factorization.
template <typename value_type>
class QR {
    public:
        QR() : m(0), n(0) {}

        void compute(unsigned rows, unsigned cols, value_type *A, bool needQ = true) {
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

            r = A;

            tau.resize(k);

            for(unsigned i = 0, ia = 0; i < k; ++i, ia += n) {
                // Generate elementary reflector H(i) to annihilate A[i+1:m)[i]
                tau[i] = gen_reflector(m-i, A[ia+i], A+ia+n+i, n);

                if (i+1 < n) {
                    // Apply H(i)' to A[i:m)[i+1:n) from the left
                    apply_reflector(m-i, n-i-1, A+ia+i, n, math::adjoint(tau[i]), A+ia+i+1, n);
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
            for(unsigned i = 0, ia = 0; i < n; ++i, ia += n)
                apply_reflector(m-i, 1, r+ia+i, n, math::adjoint(tau[i]), f+i, 1);

            std::copy(f, f+n, x);

            for(unsigned i = n; i --> 0;) {
                value_type rii = r[i*n+i];
                if (math::is_zero(rii)) continue;
                x[i] = math::inverse(rii) * x[i];

                for(unsigned j = 0, ja = 0; j < i; ++j, ja += n)
                    x[j] -= r[ja+i] * x[i];
            }
        }
    private:
        typedef typename math::scalar_of<value_type>::type scalar_type;

        static scalar_type sqr(scalar_type x) { return x * x; }

        unsigned m, n, k;

        value_type *r;
        std::vector<value_type> tau;
        std::vector<value_type> q;

        static value_type gen_reflector(int order, value_type &alpha, value_type *x, int stride) {
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
            for(int i = 0, ix = 0; i < n; ++i, ix += stride)
                xnorm2 += sqr(math::norm(x[ix]));

            if (math::is_zero(xnorm2)) return tau;

            scalar_type beta = sqrt(sqr(math::norm(alpha)) + xnorm2);

            tau = math::identity<value_type>() - math::inverse(beta) * alpha;
            alpha = math::inverse(alpha - beta * math::identity<value_type>());

            for(int i = 0, ii = 0; i < n; ++i, ii += stride)
                x[ii] = alpha * x[ii];

            alpha = beta * math::identity<value_type>();
            return tau;
        }

        static void apply_reflector(
                int m, int n, const value_type *v, int v_stride, value_type tau,
                value_type *C, int c_stride
                )
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
             *  m        The number of rows of the matrix C.
             *
             *  n        The number of columns of the matrix C.
             *
             *  v        The vector v in the representation of H.
             *           v is not used if tau = 0.
             *           The value of v[0] is ignored and assumed to be 1.
             *
             *  v_stride The increment between elements of v.
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

            // w = C` * v; C -= tau * v * w`
            for(int i = 0; i < n; ++i) {
                value_type s = math::adjoint(C[i]);
                for(int j = 1, jc = c_stride, jv = v_stride; j < m; ++j, jc += c_stride, jv += v_stride) {
                    s += math::adjoint(C[jc + i]) * v[jv];
                }

                s = tau * math::adjoint(s);
                C[i] -= s;
                for(int j = 1, jc = c_stride, jv = v_stride; j < m; ++j, jc += c_stride, jv += v_stride) {
                    C[jc + i] -= v[jv] * s;
                }
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
            q.resize(n * m);

            // Initialise columns k+1:n to zero.
            // [In the original code these were initialized to the columns of
            // the unit matrix, but since k = min(n,m), the main diagonal is
            // never seen here].
            for(unsigned j = k; j < n; ++j) {
                for(unsigned i = 0, iq = 0; i < m; ++i, iq += n)
                    q[iq + j] = math::zero<value_type>();
            }

            for(unsigned i = k; i --> 0;) {
                // Apply H(i) to A[i:m)[i+1:n) from the left
                if (i+1 < n)
                    apply_reflector(m-i, n-i-1, r+i*n+i, n, tau[i], &q[i*n+i+1], n);

                // Copy i-th reflector (including zeros and unit diagonal)
                // to the column of Q to be processed next
                unsigned ja = 0;
                for(unsigned j = 0; j < i; ++j, ja += n)
                    q[ja+i] = math::zero<value_type>();

                q[ja+i] = math::identity<value_type>() - tau[i];
                ja += n;

                for(unsigned j = i + 1; j < m; ++j, ja += n)
                    q[ja+i] = -tau[i] * r[ja+i];
            }
        }
};

} // namespace detail
} // namespace amgcl

#endif
