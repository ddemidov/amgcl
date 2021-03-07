#define BOOST_TEST_MODULE TestStaticMatrix
#include <boost/test/unit_test.hpp>

#include <amgcl/value_type/static_matrix.hpp>

BOOST_AUTO_TEST_SUITE( test_static_matrix )

BOOST_AUTO_TEST_CASE( sum ) {
    amgcl::static_matrix<int, 2, 2> a = {{1, 2, 3, 4}};
    amgcl::static_matrix<int, 2, 2> b = {{4, 3, 2, 1}};
    amgcl::static_matrix<int, 2, 2> c = a + b;

    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            BOOST_CHECK_EQUAL(c(i,j), 5);
}

BOOST_AUTO_TEST_CASE( minus ) {
    amgcl::static_matrix<int, 2, 2> a = {{5, 5, 5, 5}};
    amgcl::static_matrix<int, 2, 2> b = {{4, 3, 2, 1}};
    amgcl::static_matrix<int, 2, 2> c = a - b;

    for(int i = 0; i < 4; ++i)
        BOOST_CHECK_EQUAL(c(i), i+1);
}

BOOST_AUTO_TEST_CASE( product ) {
    amgcl::static_matrix<int, 2, 2> a = {{2, 1, 1, 2}};
    amgcl::static_matrix<int, 2, 2> c = a * a;

    BOOST_CHECK_EQUAL(c(0,0), 5);
    BOOST_CHECK_EQUAL(c(0,1), 4);
    BOOST_CHECK_EQUAL(c(1,0), 4);
    BOOST_CHECK_EQUAL(c(1,1), 5);
}

BOOST_AUTO_TEST_CASE( scale ) {
    amgcl::static_matrix<int, 2, 2> a = {{1, 2, 3, 4}};
    amgcl::static_matrix<int, 2, 2> c = 2 * a;

    for(int i = 0; i < 4; ++i)
        BOOST_CHECK_EQUAL(c(i), 2 * (i+1));
}

BOOST_AUTO_TEST_CASE( inner_product ) {
    amgcl::static_matrix<int, 2, 1> a = {{1, 2}};
    int c = amgcl::math::inner_product(a, a);

    BOOST_CHECK_EQUAL(c, 5);
}

BOOST_AUTO_TEST_CASE( inverse ) {
    amgcl::static_matrix<double, 2, 2> a = {{2.0, -1.0, -1.0, 2.0}};
    amgcl::static_matrix<double, 2, 2> b = amgcl::math::inverse(a);
    amgcl::static_matrix<double, 2, 2> c = b * a;

    for(int i = 0; i < 2; ++i)
        for(int j = 0; j < 2; ++j)
            BOOST_CHECK_SMALL(c(i,j) - (i == j), 1e-8);
}

BOOST_AUTO_TEST_CASE( inverse_pivoting ) {
    amgcl::static_matrix<double, 4, 4> a {{
    1, -0.1, -0.028644256, 0.25684664,
    1, -0.1, -0.025972342, 0.25663863,
    1, -0.095699158, -0.029327056, 0.25554974,
    1, -0.09543351, -0.026189496, 0.25796741,
    }};
    amgcl::static_matrix<double, 4, 4> b = amgcl::math::inverse(a);
    amgcl::static_matrix<double, 4, 4> c = b * a;

    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            BOOST_CHECK_SMALL(c(i,j) - (i == j), 1e-8);
}

BOOST_AUTO_TEST_CASE( inverse_pivoting_2 ) {
    amgcl::static_matrix<double, 4, 4> a {{
    0, 1, 0, 0,
    0, 0, 1, 0,
    1, 0, 0, 0,
    0, 0, 0, 1,
    }};
    amgcl::static_matrix<double, 4, 4> b = amgcl::math::inverse(a);
    amgcl::static_matrix<double, 4, 4> c = b * a;

    for(int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            BOOST_CHECK_SMALL(c(i,j) - (i == j), 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()
