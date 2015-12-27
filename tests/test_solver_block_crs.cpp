#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/block_crs.hpp>

#include "test_solver.hpp"

BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE(test_block_crs_backend)
{
    test_backend< amgcl::backend::block_crs<double> >();
}

BOOST_AUTO_TEST_SUITE_END()
