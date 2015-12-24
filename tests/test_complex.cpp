#define BOOST_TEST_MODULE TestComplex
#include <boost/test/unit_test.hpp>

#include <complex>

#include <amgcl/value_type/complex.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/amgcl.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/complex.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

BOOST_AUTO_TEST_SUITE( test_complex )

BOOST_AUTO_TEST_CASE(complex_matrix)
{
    typedef std::complex<double> complex;

    std::vector<int>     ptr;
    std::vector<int>     col;
    std::vector<complex> val;
    std::vector<complex> rhs;

    size_t n = sample_problem(32, val, col, ptr, rhs);

    std::vector<complex> x(n, amgcl::math::make_zero<complex>());

    typedef amgcl::backend::builtin<complex> Backend;

    amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::gauss_seidel
            >,
        amgcl::solver::gmres<Backend>
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

BOOST_AUTO_TEST_CASE(complex_matrix_adapter)
{
    typedef std::complex<double> complex;

    std::vector<int>     ptr;
    std::vector<int>     col;
    std::vector<complex> val;
    std::vector<complex> rhs;

    size_t n = sample_problem(32, val, col, ptr, rhs);

    std::vector<complex> x(n, complex(0.0,0.0));

    typedef amgcl::backend::builtin<double> Backend;

    boost::property_tree::ptree prm;
    prm.put("precond.coarsening.aggr.block_size", 2);

    amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<Backend>
        > solve( amgcl::adapter::complex_matrix(boost::tie(n, ptr, col, val)), prm );

    std::cout << solve.precond() << std::endl;

    boost::iterator_range<const double*> f_range =
        amgcl::adapter::complex_range(rhs);

    boost::iterator_range<double*> x_range =
        amgcl::adapter::complex_range(x);

    size_t iters;
    double resid;

    boost::tie(iters, resid) = solve(f_range, x_range);

    BOOST_CHECK_SMALL(resid, 1e-8);

    std::cout
        << "iters: " << iters << std::endl
        << "resid: " << resid << std::endl
        ;
}

BOOST_AUTO_TEST_SUITE_END()
