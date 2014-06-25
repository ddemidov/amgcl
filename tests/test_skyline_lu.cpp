#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <amgcl/solver/skyline_lu.hpp>
#include <amgcl/backend/builtin.hpp>
#include "sample_problem.hpp"

BOOST_AUTO_TEST_SUITE( test_skyline_lu )

BOOST_AUTO_TEST_CASE(skyline_lu)
{
    amgcl::backend::crs<double> A;
    std::vector<double> rhs;

    size_t n = A.nrows = A.ncols = sample_problem(16, A.val, A.col, A.ptr, rhs);

    amgcl::solver::skyline_lu<double> solve( A );

    std::vector<double> x(n);
    std::vector<double> r(n);

    solve(rhs, x);

    amgcl::backend::residual(rhs, A, x, r);

    BOOST_CHECK_SMALL(sqrt(amgcl::backend::inner_product(r, r)), 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()

