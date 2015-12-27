#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <amgcl/backend/viennacl.hpp>

#include "test_solver.hpp"

BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE(test_blaze_backend)
{
    test_backend< amgcl::backend::viennacl< viennacl::compressed_matrix<double> > >();
    test_backend< amgcl::backend::viennacl< viennacl::ell_matrix<double> > >();
    test_backend< amgcl::backend::viennacl< viennacl::hyb_matrix<double> > >();
}

BOOST_AUTO_TEST_SUITE_END()
