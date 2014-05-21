#define BOOST_TEST_MODULE MatrixEigen
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/eigen.hpp>

BOOST_AUTO_TEST_SUITE( backend_eigen )

BOOST_AUTO_TEST_CASE(iteration)
{
    using namespace amgcl::backend;

    // [1 2 0 0]
    // [0 3 4 0]
    // [5 0 6 0]
    // [0 0 0 7]
    int   ptr[] = {0, 2, 4, 6, 7};
    int   col[] = {0, 1, 1, 2, 0, 2, 3};
    float val[] = {1, 2, 3, 4, 5, 6, 7};

    typedef Eigen::MappedSparseMatrix<float, Eigen::RowMajor, int> Matrix;
    Matrix A(4, 4, 7, ptr, col, val);

    BOOST_CHECK_EQUAL( 4, nrows(A) );
    BOOST_CHECK_EQUAL( 4, ncols(A) );

    typedef row_iterator<Matrix>::type row_iterator;
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
