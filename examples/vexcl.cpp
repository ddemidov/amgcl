#include <iostream>

#include <amgcl/amgcl.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

int main(int argc, char *argv[]) {
    using amgcl::prof;

    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    prof.tic("assemble");
    int m = argc > 1 ? std::stoi(argv[1]) : 128;
    int n = sample_problem(m, val, col, ptr, rhs);
    prof.toc("assemble");

    typedef amgcl::backend::vexcl<double> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab< Backend >
        > Solver;

    Solver::params sprm;
    Backend::params bprm;
    bprm.q = ctx;

    prof.tic("build");
    Solver solve( boost::tie(n, ptr, col, val), sprm, bprm );
    prof.toc("build");

    std::cout << solve.precond() << std::endl;

    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(ctx, n);
    x = 0;

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(f, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << prof << std::endl;
}
