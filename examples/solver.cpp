#include <iostream>

#include <amgcl/amgcl.hpp>

#include <amgcl/backend/crs_tuple.hpp>
#include <amgcl/backend/block_crs.hpp>
#include <amgcl/backend/eigen.hpp>

#include <amgcl/coarsening/aggregation.hpp>

#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>

#include <amgcl/solver/cg.hpp>

#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof("v2");
}

int main() {
    using amgcl::prof;

    typedef amgcl::amg<
        amgcl::backend::block_crs<double>,
        amgcl::coarsening::aggregation,
        amgcl::relaxation::damped_jacobi
        > AMG;

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    prof.tic("assemble");
    int n = sample_problem(128, val, col, ptr, rhs);
    prof.toc("assemble");

    prof.tic("build");
    AMG amg( boost::tie(n, n, val, col, ptr) );
    prof.toc("build");

    std::cout << amg << std::endl;

    boost::shared_ptr<AMG::vector> f = AMG::backend_type::copy_vector(rhs, amg.prm.backend);
    boost::shared_ptr<AMG::vector> x = AMG::backend_type::create_vector(n, amg.prm.backend);

    amgcl::backend::clear(*x);

    amgcl::solver::cg<AMG::backend_type> solve(n);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(amg.top_matrix(), *f, amg, *x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << amgcl::prof << std::endl;
}
