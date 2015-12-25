#define BOOST_TEST_MODULE TestQR
#include <boost/test/unit_test.hpp>

#include <vector>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>

#include <amgcl/detail/qr.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/value_type/complex.hpp>

template <class value_type>
void run_qr_test() {
    const size_t n = 5;
    const size_t m = 3;

    boost::random::mt19937 gen;
    boost::random::uniform_real_distribution<value_type> rnd;

    boost::multi_array<value_type, 2> A0(boost::extents[n][m]);

    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < m; ++j)
            A0[i][j] = rnd(gen);

    boost::multi_array<value_type, 2> A = A0;

    amgcl::detail::QR<value_type> qr;

    qr.compute(n, m, A.data());

    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            value_type sum = amgcl::math::zero<value_type>();

            for(size_t k = 0; k < m; ++k)
                sum += qr.Q(i,k) * qr.R(k,j);

            BOOST_CHECK_SMALL(A0[i][j] - sum, 1e-8);
        }
    }
}

BOOST_AUTO_TEST_SUITE( test_qr )

BOOST_AUTO_TEST_CASE( test_qr ) {
    run_qr_test<double>();
    // TODO: run_qr_test< std::complex<double> >();
}

BOOST_AUTO_TEST_SUITE_END()
