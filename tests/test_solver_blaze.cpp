#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/blaze.hpp>

#include "test_solver.hpp"

BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE(test_blaze_backend)
{
    test_backend< amgcl::backend::blaze<double> >();
}

BOOST_AUTO_TEST_SUITE_END()
