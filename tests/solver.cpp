#define BOOST_TEST_MODULE Solver
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <amgcl/backend/crs_tuple.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/builder.hpp>
#include <amgcl/tictoc.hpp>
#include "sample_matrix.hpp"

namespace amgcl {
    profiler<> prof("v2");
}

BOOST_AUTO_TEST_SUITE( test_solver )

BOOST_AUTO_TEST_CASE(build)
{
    typedef amgcl::builder< double, amgcl::coarsening::aggregation > builder;

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;

    TIC("assemble");
    int n = sample_matrix(128, val, col, ptr);
    TOC("assemble")

    TIC("build");
    builder amg( boost::tie(n, n, val, col, ptr) );
    TOC("build");

    std::cout << amg << std::endl << amgcl::prof << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
