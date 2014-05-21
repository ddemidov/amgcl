#define BOOST_TEST_MODULE MatrixCCRS
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/ccrs.hpp>

BOOST_AUTO_TEST_SUITE( backend_ccrs )

BOOST_AUTO_TEST_CASE(construct)
{
    BOOST_REQUIRE_NO_THROW(
            amgcl::backend::ccrs<float> A(4, 4)
            );
}

BOOST_AUTO_TEST_CASE(build)
{
    using namespace amgcl::backend;

    // A: [1 2 0 0]
    //    [0 3 4 0]
    //    [5 0 6 0]
    //    [0 0 0 7]
    const int   ptr[] = {0, 2, 4, 6, 7};
    const int   col[] = {0, 1, 1, 2, 0, 2, 3};
    const float val[] = {1, 2, 3, 4, 5, 6, 7};

    typedef ccrs<float> Matrix;
    Matrix A(4, 4);

    A.insert(0, 4, ptr, col, val);
    A.finalize();

    BOOST_CHECK_EQUAL(4, nrows(A));
    BOOST_CHECK_EQUAL(4, ncols(A));

    typedef typename row_iterator<Matrix>::type row_iterator;
    for(int row = 0; row < 4; ++row) {
        int j = ptr[row];
        for(row_iterator it = row_begin(A, row); it; ++it, ++j) {
            BOOST_REQUIRE(j < ptr[row + 1]);
            BOOST_CHECK_EQUAL( col[j], it.col() );
            BOOST_CHECK_EQUAL( val[j], it.value() );
        }
    }
}

BOOST_AUTO_TEST_CASE(transpose)
{
    using namespace amgcl::backend;

    // A: [1 2 0 0]
    //    [0 3 4 0]
    //    [5 0 6 0]
    //    [0 0 0 7]
    //
    // B: [1 0 5 0]
    //    [2 3 0 0]
    //    [0 4 6 0]
    //    [0 0 0 7]
    const int   ptr_a[] = {0, 2, 4, 6, 7};
    const int   col_a[] = {0, 1, 1, 2, 0, 2, 3};
    const float val_a[] = {1, 2, 3, 4, 5, 6, 7};

    const int   ptr_b[] = {0, 2, 4, 6, 7};
    const int   col_b[] = {0, 2, 0, 1, 1, 2, 3};
    const float val_b[] = {1, 5, 2, 3, 4, 6, 7};

    typedef ccrs<float> Matrix;
    Matrix A(4, 4);

    A.insert(0, 4, ptr_a, col_a, val_a);
    A.finalize();

    Matrix B = transp(A);

    BOOST_CHECK_EQUAL(4, nrows(B));
    BOOST_CHECK_EQUAL(4, ncols(B));

    typedef typename row_iterator<Matrix>::type row_iterator;
    for(int row = 0; row < 4; ++row) {
        int j = ptr_b[row];
        for(row_iterator it = row_begin(B, row); it; ++it, ++j) {
            BOOST_REQUIRE(j < ptr_b[row + 1]);
            BOOST_CHECK_EQUAL( col_b[j], it.col() );
            BOOST_CHECK_EQUAL( val_b[j], it.value() );
        }
    }
}

BOOST_AUTO_TEST_CASE(multiply)
{
    using namespace amgcl::backend;

    // A: [1 2 0 0]
    //    [0 3 4 0]
    //    [5 0 6 0]
    //    [0 0 0 7]
    //
    // B: [1 0 5 0]
    //    [2 3 0 0]
    //    [0 4 6 0]
    //    [0 0 0 7]
    //
    // C: [5  6  5  0]
    //    [6 25 24  0]
    //    [5 24 61  0]
    //    [0  0  0 49]

    const int   ptr_a[] = {0, 2, 4, 6, 7};
    const int   col_a[] = {0, 1, 1, 2, 0, 2, 3};
    const float val_a[] = {1, 2, 3, 4, 5, 6, 7};

    const int   ptr_c[] = {0, 3, 6, 9, 10};
    const int   col_c[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 3};
    const float val_c[] = {5, 6, 5, 6, 25, 24, 5, 24, 61, 49};

    typedef ccrs<float> Matrix;
    Matrix A(4, 4);

    A.insert(0, 4, ptr_a, col_a, val_a);
    A.finalize();

    Matrix B = prod(A, transp(A));

    BOOST_CHECK_EQUAL(4, nrows(B));
    BOOST_CHECK_EQUAL(4, ncols(B));

    typedef typename row_iterator<Matrix>::type row_iterator;
    for(int row = 0; row < 4; ++row) {
        int j = ptr_c[row];
        for(row_iterator it = row_begin(B, row); it; ++it, ++j) {
            BOOST_REQUIRE(j < ptr_c[row + 1]);
            BOOST_CHECK_EQUAL( col_c[j], it.col() );
            BOOST_CHECK_EQUAL( val_c[j], it.value() );
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
