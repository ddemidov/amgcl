#define BOOST_TEST_MODULE TestQR
#include <boost/test/unit_test.hpp>

#include <vector>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>

#include <amgcl/detail/qr.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/value_type/complex.hpp>

#ifdef AMGCL_HAVE_EIGEN
#  include <amgcl/value_type/eigen.hpp>
#endif

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

#ifdef AMGCL_HAVE_EIGEN
template <class T, int N, int M>
struct make_random< Eigen::Matrix<T,N,M> > {
    static Eigen::Matrix<T,N,M> get() {
        return Eigen::Matrix<T,N,M>::Random();
    }
};
#endif

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
    std::vector<value_type> f0(n, amgcl::math::identity<value_type>());
    std::vector<value_type> f = f0;

    std::vector<value_type> x(m);

    qr.solve(f.data(), x.data());

    std::vector<value_type> Ax(n);
    for(size_t i = 0; i < n; ++i) {
        value_type sum = amgcl::math::zero<value_type>();
        for(size_t j = 0; j < m; ++j)
            sum += A0[i][j] * x[j];

        Ax[i] = sum;
    }

    for(size_t i = 0; i < m; ++i) {
        value_type sumx = amgcl::math::zero<value_type>();
        value_type sumf = amgcl::math::zero<value_type>();

        for(size_t j = 0; j < n; ++j) {
            sumx += amgcl::math::adjoint(A0[j][i]) * Ax[j];
            sumf += amgcl::math::adjoint(A0[j][i]) * f0[j];
        }

        value_type delta = sumx - sumf;

        BOOST_CHECK_SMALL(amgcl::math::norm(delta), 1e-8);
    }
}

BOOST_AUTO_TEST_SUITE( test_qr )

BOOST_AUTO_TEST_CASE( test_qr ) {
    run_qr_test<double>();
    run_qr_test< std::complex<double> >();
    //run_qr_test< Eigen::Matrix<double, 2, 2> >();
}

BOOST_AUTO_TEST_SUITE_END()
