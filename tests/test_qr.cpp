#define BOOST_TEST_MODULE TestQR
#include <boost/test/unit_test.hpp>

#include <vector>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>

#include <amgcl/detail/qr.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/value_type/complex.hpp>
#include <amgcl/value_type/static_matrix.hpp>

template <class T>
struct make_random {
    static T get() {
        static boost::random::mt19937 gen;
        static boost::random::uniform_real_distribution<T> rnd;

        return rnd(gen);
    }
};

template <class T>
T random() {
    return make_random<T>::get();
}

template <class T>
struct make_random< std::complex<T> > {
    static std::complex<T> get() {
        return std::complex<T>( random<T>(), random<T>() );
    }
};

template <class T, int N, int M>
struct make_random< amgcl::static_matrix<T,N,M> > {
    typedef amgcl::static_matrix<T,N,M> matrix;
    static matrix get() {
        matrix A = amgcl::math::zero<matrix>();
        for(int i = 0; i < N; ++i)
            for(int j = 0; j < M; ++j)
                A(i,j) = make_random<T>::get();
        return A;
    }
};

namespace amgcl { namespace detail {

template <typename T, int N, int M>
class QR< amgcl::static_matrix<T,N,M> > {
    public:
        typedef amgcl::static_matrix<T,N,M> value_type;
        typedef typename amgcl::math::rhs_of<value_type>::type rhs_type;

        QR() {}

        void compute(unsigned rows, unsigned cols, value_type *A, bool needQ = true) {
            buf.resize(rows * cols * N * M);

            for(unsigned i = 0, ix=0; i < rows; ++i)
                for(int ii = 0; ii < N; ++ii)
                    for(unsigned j = 0; j < cols; ++j)
                        for(int jj = 0; jj < M; ++jj, ++ix)
                            buf[ix] = A[i * cols + j](ii, jj);

            base.compute(rows * N, cols * M, buf.data(), needQ);
        }

        value_type R(unsigned i, unsigned j) const {
            value_type v;
            if (j < i) {
                v = math::zero<value_type>();
            } else {
                for(int ii = 0; ii < N; ++ii)
                    for(int jj = 0; jj < M; ++jj)
                        v(ii,jj) = base.R(i * N + ii, j * M + jj);
            }
            return v;
        }

        // Returns element of the matrix Q.
        value_type Q(unsigned i, unsigned j) const {
            value_type v;
            for(int ii = 0; ii < N; ++ii)
                for(int jj = 0; jj < M; ++jj)
                    v(ii,jj) = base.Q(i * N + ii, j * M + jj);
            return v;
        }

        // Solves the system Q R x = f
        void solve(rhs_type *f, rhs_type *x) const {
            base.solve(reinterpret_cast<T*>(f), reinterpret_cast<T*>(x));
        }

        QR<T> base;
        std::vector<T> buf;
};

} }

template <class value_type>
void run_qr_test() {
    const size_t n = 5;
    const size_t m = 3;

    boost::multi_array<value_type, 2> A0(boost::extents[n][m]);

    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < m; ++j)
            A0[i][j] = random<value_type>();

    boost::multi_array<value_type, 2> A = A0;

    amgcl::detail::QR<value_type> qr;

    qr.compute(n, m, A.data());

    // Check that A = QR
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            value_type sum = amgcl::math::zero<value_type>();

            for(size_t k = 0; k < m; ++k)
                sum += qr.Q(i,k) * qr.R(k,j);

            sum -= A0[i][j];

            BOOST_CHECK_SMALL(amgcl::math::norm(sum), 1e-8);
        }
    }

    // Check that solution works (A^t A x == A^t f).
    typedef typename amgcl::math::rhs_of<value_type>::type rhs_type;
    std::vector<rhs_type> f0(n, amgcl::math::constant<rhs_type>(1));
    std::vector<rhs_type> f = f0;

    std::vector<rhs_type> x(m);

    qr.solve(f.data(), x.data());

    std::vector<rhs_type> Ax(n);
    for(size_t i = 0; i < n; ++i) {
        rhs_type sum = amgcl::math::zero<rhs_type>();
        for(size_t j = 0; j < m; ++j)
            sum += A0[i][j] * x[j];

        Ax[i] = sum;
    }

    for(size_t i = 0; i < m; ++i) {
        rhs_type sumx = amgcl::math::zero<rhs_type>();
        rhs_type sumf = amgcl::math::zero<rhs_type>();

        for(size_t j = 0; j < n; ++j) {
            sumx += amgcl::math::adjoint(A0[j][i]) * Ax[j];
            sumf += amgcl::math::adjoint(A0[j][i]) * f0[j];
        }

        rhs_type delta = sumx - sumf;

        BOOST_CHECK_SMALL(amgcl::math::norm(delta), 1e-8);
    }
}

BOOST_AUTO_TEST_SUITE( test_qr )

BOOST_AUTO_TEST_CASE( test_qr ) {
    run_qr_test<double>();
    run_qr_test< std::complex<double> >();
    run_qr_test< amgcl::static_matrix<double, 2, 2> >();
}

BOOST_AUTO_TEST_SUITE_END()
