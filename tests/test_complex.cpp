#define BOOST_TEST_MODULE TestComplex
#include <boost/test/unit_test.hpp>

#include <complex>

#include <amgcl/backend/enable_complex.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/amgcl.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
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

    std::vector<complex> x(n, complex(0.0,0.0));

    typedef amgcl::backend::builtin<complex> Backend;

    amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<Backend>
        > solve( boost::tie(n, ptr, col, val) );
}

BOOST_AUTO_TEST_SUITE_END()
