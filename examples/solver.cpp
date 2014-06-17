#include <iostream>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/crs_tuple.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

int main(int argc, char *argv[]) {
    amgcl::profiler<> prof;

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    prof.tic("assemble");
    int m = argc > 1 ? atoi(argv[1]) : 128;
    int n = sample_problem(m, val, col, ptr, rhs);
    prof.toc("assemble");

    prof.tic("build");
    amgcl::make_solver<
        amgcl::backend::builtin<double>,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::spai0,
        amgcl::solver::bicgstab
        > solve( boost::tie(n, ptr, col, val) );
    prof.toc("build");

    std::cout << solve << std::endl;

    std::vector<double> x(n, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(rhs, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << prof << std::endl;
}
