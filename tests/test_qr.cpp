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
void qr_factorize(int n, int m) {
    std::cout << "factorize " << n << " " << m << std::endl;
    typedef typename boost::conditional<order == amgcl::detail::row_major,
            boost::c_storage_order,
            boost::fortran_storage_order
            >::type ma_storage_order;

    boost::multi_array<value_type, 2> A0(boost::extents[n][m], ma_storage_order());

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            A0[i][j] = random<value_type>();

    boost::multi_array<value_type, 2> A = A0;

    amgcl::detail::QR<value_type> qr;

    qr.factorize(n, m, A.data(), order);

    // Check that A = QR
    int p = std::min(n, m);
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            value_type sum = amgcl::math::zero<value_type>();

            for(int k = 0; k < p; ++k)
                sum += qr.Q(i,k) * qr.R(k,j);

            sum -= A0[i][j];

            BOOST_CHECK_SMALL(amgcl::math::norm(sum), 1e-8);
        }
    }
}

template <class value_type, amgcl::detail::storage_order order>
void qr_solve(int n, int m) {
    std::cout << "solve " << n << " " << m << std::endl;
    typedef typename boost::conditional<order == amgcl::detail::row_major,
            boost::c_storage_order,
            boost::fortran_storage_order
            >::type ma_storage_order;

    typedef typename amgcl::math::rhs_of<value_type>::type rhs_type;

    boost::multi_array<value_type, 2> A0(boost::extents[n][m], ma_storage_order());

    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            A0[i][j] = random<value_type>();

    boost::multi_array<value_type, 2> A = A0;

    amgcl::detail::QR<value_type> qr;

    std::vector<rhs_type> f0(n, amgcl::math::constant<rhs_type>(1));
    std::vector<rhs_type> f = f0;

    std::vector<rhs_type> x(m);

    qr.solve(n, m, A.data(), f.data(), x.data(), order);

    std::vector<rhs_type> Ax(n);
    for(int i = 0; i < n; ++i) {
        rhs_type sum = amgcl::math::zero<rhs_type>();
        for(int j = 0; j < m; ++j)
            sum += A0[i][j] * x[j];

        Ax[i] = sum;

        if (n < m) {
            BOOST_CHECK_SMALL(amgcl::math::norm(sum - f0[i]), 1e-8);
        }
    }

    if (n >= m) {
        for(int i = 0; i < m; ++i) {
            rhs_type sumx = amgcl::math::zero<rhs_type>();
            rhs_type sumf = amgcl::math::zero<rhs_type>();

            for(int j = 0; j < n; ++j) {
                sumx += amgcl::math::adjoint(A0[j][i]) * Ax[j];
                sumf += amgcl::math::adjoint(A0[j][i]) * f0[j];
            }

            rhs_type delta = sumx - sumf;

            BOOST_CHECK_SMALL(amgcl::math::norm(delta), 1e-8);
        }
    }
}

BOOST_AUTO_TEST_SUITE( test_qr )

BOOST_AUTO_TEST_CASE( test_qr_factorize ) {
    const int shape[][2] = {
        {3, 3},
        {3, 5},
        {5, 3},
        {5, 5}
    };

    const int n = sizeof(shape) / sizeof(shape[0]);

    for(int i = 0; i < n; ++i) {
        qr_factorize<double,                             amgcl::detail::row_major>(shape[i][0], shape[i][1]);
        qr_factorize<double,                             amgcl::detail::col_major>(shape[i][0], shape[i][1]);
        qr_factorize<std::complex<double>,               amgcl::detail::row_major>(shape[i][0], shape[i][1]);
        qr_factorize<std::complex<double>,               amgcl::detail::col_major>(shape[i][0], shape[i][1]);
        qr_factorize<amgcl::static_matrix<double, 2, 2>, amgcl::detail::row_major>(shape[i][0], shape[i][1]);
        qr_factorize<amgcl::static_matrix<double, 2, 2>, amgcl::detail::col_major>(shape[i][0], shape[i][1]);
    }
}

BOOST_AUTO_TEST_CASE( test_qr_solve ) {
    const int shape[][2] = {
        {3, 3},
        {3, 5},
        {5, 3},
        {5, 5}
    };

    const int n = sizeof(shape) / sizeof(shape[0]);

    for(int i = 0; i < n; ++i) {
        qr_solve<double,                             amgcl::detail::row_major>(shape[i][0], shape[i][1]);
        qr_solve<double,                             amgcl::detail::col_major>(shape[i][0], shape[i][1]);
        qr_solve<std::complex<double>,               amgcl::detail::row_major>(shape[i][0], shape[i][1]);
        qr_solve<std::complex<double>,               amgcl::detail::col_major>(shape[i][0], shape[i][1]);
        qr_solve<amgcl::static_matrix<double, 2, 2>, amgcl::detail::row_major>(shape[i][0], shape[i][1]);
        qr_solve<amgcl::static_matrix<double, 2, 2>, amgcl::detail::col_major>(shape[i][0], shape[i][1]);
    }
}

BOOST_AUTO_TEST_CASE( qr_issue_39 ) {
    boost::multi_array<double, 2> A0(boost::extents[2][2]);
    A0[0][0] = 1e+0;
    A0[0][1] = 1e+0;
    A0[1][0] = 1e-8;
    A0[1][1] = 1e+0;

    boost::multi_array<double, 2> A = A0;

    amgcl::detail::QR<double> qr;

    qr.factorize(2, 2, A.data());

    // Check that A = QR
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            double sum = 0;
            for(int k = 0; k < 2; ++k)
                sum += qr.Q(i,k) * qr.R(k,j);

            sum -= A0[i][j];

            BOOST_CHECK_SMALL(sum, 1e-8);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
