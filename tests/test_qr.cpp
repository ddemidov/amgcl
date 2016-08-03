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

template <class value_type, amgcl::detail::storage_order order>
void run_qr_test() {
    const size_t n = 5;
    const size_t m = 3;

    typedef typename boost::conditional<order == amgcl::detail::row_major,
            boost::c_storage_order,
            boost::fortran_storage_order
            >::type ma_storage_order;

    boost::multi_array<value_type, 2> A0(boost::extents[n][m], ma_storage_order());

    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < m; ++j)
            A0[i][j] = random<value_type>();

    boost::multi_array<value_type, 2> A = A0;

    amgcl::detail::QR<value_type, order> qr;

    qr.compute(n, m, A.data());
    qr.compute_q();

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
    run_qr_test< double,                             amgcl::detail::row_major>();
    run_qr_test< double,                             amgcl::detail::col_major>();
    run_qr_test< std::complex<double>,               amgcl::detail::row_major>();
    run_qr_test< std::complex<double>,               amgcl::detail::col_major>();
    run_qr_test< amgcl::static_matrix<double, 2, 2>, amgcl::detail::row_major>();
    run_qr_test< amgcl::static_matrix<double, 2, 2>, amgcl::detail::col_major>();
}

BOOST_AUTO_TEST_SUITE_END()
