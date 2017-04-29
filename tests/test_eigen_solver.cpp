#define BOOST_TEST_MODULE TestEigenSolver
#include <boost/test/unit_test.hpp>

#include <Eigen/SparseLU>
#include <amgcl/solver/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include "sample_problem.hpp"

BOOST_AUTO_TEST_SUITE( test_eigen_solver )

BOOST_AUTO_TEST_CASE(eigen_solver)
{
    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    size_t n = sample_problem(16, val, col, ptr, rhs);

    amgcl::backend::crs<double, int> A;

    A.nrows = A.ncols = n;
    A.nnz = ptr[n];
    A.ptr = ptr.data();
    A.col = col.data();
    A.val = val.data();
    A.own_data = false;

    amgcl::solver::EigenSolver<Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, int> > > solve( A );

    std::vector<double> x(n);
    std::vector<double> r(n);

    solve(rhs, x);

    amgcl::backend::residual(rhs, A, x, r);

    BOOST_CHECK_SMALL(sqrt(amgcl::backend::inner_product(r, r)), 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()

