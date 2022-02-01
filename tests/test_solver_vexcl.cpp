#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/vexcl.hpp>

#include "test_solver.hpp"

BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE(test_vexcl_backend)
{
    vex::Context ctx(vex::Filter::Env && vex::Filter::DoublePrecision && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

    typedef amgcl::backend::vexcl<double> Backend;
    Backend::params bprm;
    bprm.q = ctx;

    test_backend< amgcl::backend::vexcl< double > >(bprm);
    test_backend< amgcl::backend::vexcl< double, int, ptrdiff_t > >(bprm);
    test_backend< amgcl::backend::vexcl< double, int, int > >(bprm);
    test_backend< amgcl::backend::vexcl< double, uint32_t, size_t > >(bprm);
}

BOOST_AUTO_TEST_SUITE_END()
