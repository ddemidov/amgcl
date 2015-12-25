#define BOOST_TEST_MODULE TestComplex
#include <boost/test/unit_test.hpp>

#include <amgcl/value_type/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/amgcl.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/complex.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

BOOST_AUTO_TEST_SUITE( test_eigen_values )

BOOST_AUTO_TEST_CASE(eigen_value_type)
{
    typedef Eigen::Matrix<double, 2, 2> M2;
    typedef Eigen::Matrix<double, 2, 1> V2;

    std::vector<int> ptr;
    std::vector<int> col;
    std::vector<M2>  val;
    std::vector<V2>  rhs;

    size_t n = sample_problem(32, val, col, ptr, rhs);

    std::vector<V2> x(n, amgcl::math::zero<V2>());

    typedef amgcl::backend::builtin<M2> Backend;

    amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::gauss_seidel
            >,
        amgcl::solver::bicgstabl<Backend>
        > solve( boost::tie(n, ptr, col, val) );

    std::cout << solve.precond() << std::endl;

    size_t iters;
    double resid;

    boost::tie(iters, resid) = solve(rhs, x);

    BOOST_CHECK_SMALL(resid, 1e-8);

    std::cout
        << "iters: " << iters << std::endl
        << "resid: " << resid << std::endl
        ;
}

BOOST_AUTO_TEST_SUITE_END()
