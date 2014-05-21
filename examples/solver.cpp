#include <iostream>
#include <amgcl/backend/crs_tuple.hpp>
#include <amgcl/backend/block_crs.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/builder.hpp>
#include <amgcl/solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/profiler.hpp>
#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof("v2");
}

int main() {
    using amgcl::prof;

    typedef amgcl::backend::block_crs<double> backend;

    typedef amgcl::builder<
        double, amgcl::coarsening::aggregation
        > Builder;

    typedef amgcl::amg<
        backend, amgcl::relaxation::damped_jacobi
        > AMG;

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    backend::vector     rhs;

    prof.tic("assemble");
    int n = sample_problem(128, val, col, ptr, rhs);
    prof.toc("assemble");

    prof.tic("build");
    Builder builder( boost::tie(n, n, val, col, ptr) );
    prof.toc("build");

    std::cout << builder << std::endl;

    prof.tic("move");
    AMG amg(builder);
    prof.toc("move");

    backend::vector x(n);
    amgcl::backend::clear(x);

    amgcl::solver::cg<backend> solve(n);
    size_t iters;
    double resid;

    prof.tic("solve");
    boost::tie(iters, resid) = solve(amg.top_matrix(), rhs, amg, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << amgcl::prof << std::endl;
}
