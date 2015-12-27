#define BOOST_TEST_MODULE TestComplex
#include <boost/test/unit_test.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/eigen.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/make_solver.hpp>

#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggr_emin.hpp>

#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/parallel_ilu0.hpp>
#include <amgcl/relaxation/ilut.hpp>

#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/gmres.hpp>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

BOOST_AUTO_TEST_SUITE( test_eigen_values )

typedef Eigen::Matrix<double, 2, 2> M2;
typedef Eigen::Matrix<double, 2, 1> V2;
typedef amgcl::backend::builtin<M2> Backend;

template <
    class Coarsening,
    template <class> class Relaxation,
    class Solver
    >
void run_test(
        size_t n,
        std::vector<int> ptr,
        std::vector<int> col,
        std::vector<M2>  val,
        std::vector<V2>  rhs
        )
{
    std::vector<V2> x(n, amgcl::math::zero<V2>());

    amgcl::make_solver<
        amgcl::amg<Backend, Coarsening, Relaxation>, Solver
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

BOOST_AUTO_TEST_CASE(eigen_value_type)
{
    std::vector<int> ptr;
    std::vector<int> col;
    std::vector<M2>  val;
    std::vector<V2>  rhs;

    size_t n = sample_problem(32, val, col, ptr, rhs);

    // Solvers
    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::bicgstab<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::bicgstabl<Backend>
        >(n, ptr, col, val, rhs);

#if 0
    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::gmres<Backend>
        >(n, ptr, col, val, rhs);
#endif

    // Relaxations
    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::damped_jacobi,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::ilu0,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::parallel_ilu0,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::ilut,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::chebyshev,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    // Coarsening
    run_test<
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::aggregation,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);

    run_test<
        amgcl::coarsening::smoothed_aggr_emin,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg<Backend>
        >(n, ptr, col, val, rhs);
}

BOOST_AUTO_TEST_SUITE_END()
