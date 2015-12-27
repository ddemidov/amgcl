#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/eigen.hpp>

#include "test_solver.hpp"

BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE(test_eigen_backend)
{
    test_backend< amgcl::backend::eigen<double> >();
}

BOOST_AUTO_TEST_SUITE_END()
