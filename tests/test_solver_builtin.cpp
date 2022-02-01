#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/builtin.hpp>

#include "test_solver.hpp"

BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE(test_builtin_backend)
{
    test_backend< amgcl::backend::builtin<double> >();
    test_backend< amgcl::backend::builtin<double, int, ptrdiff_t> >();
    test_backend< amgcl::backend::builtin<double, int, int> >();
    test_backend< amgcl::backend::builtin<double, uint32_t, size_t> >();
}

BOOST_AUTO_TEST_SUITE_END()
