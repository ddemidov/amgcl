#define BOOST_TEST_MODULE MatrixCRS
#include <boost/test/unit_test.hpp>
#include <boost/range/iterator_range.hpp>
#include <amgcl/backend/crs.hpp>

BOOST_AUTO_TEST_SUITE( backend_crs )

BOOST_AUTO_TEST_CASE(construct)
{
    // [1 2 0 0]
    // [0 3 4 0]
    // [5 0 6 0]
    // [0 0 0 7]
    const int   ptr[] = {0, 2, 4, 6, 7};
    const int   col[] = {0, 1, 1, 2, 0, 2, 3};
    const float val[] = {1, 2, 3, 4, 5, 6, 7};

    BOOST_REQUIRE_NO_THROW(
            amgcl::backend::crs<float> A(
                4, 4,
                boost::make_iterator_range(ptr, ptr + 5),
                boost::make_iterator_range(col, col + 7),
                boost::make_iterator_range(val, val + 7)
                )
            );
}

BOOST_AUTO_TEST_CASE(iteration)
{
    using namespace amgcl::backend;

    // [1 2 0 0]
    // [0 3 4 0]
    // [5 0 6 0]
    // [0 0 0 7]
    const int   ptr[] = {0, 2, 4, 6, 7};
    const int   col[] = {0, 1, 1, 2, 0, 2, 3};
    const float val[] = {1, 2, 3, 4, 5, 6, 7};

    typedef crs<float> Matrix;

    Matrix A(
            4, 4,
            boost::make_iterator_range(ptr, ptr + 5),
            boost::make_iterator_range(col, col + 7),
            boost::make_iterator_range(val, val + 7)
            );

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

BOOST_AUTO_TEST_SUITE_END()
