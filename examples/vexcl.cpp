#include <iostream>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

int main(int argc, char *argv[]) {
    amgcl::profiler<> prof;

    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;

    typedef amgcl::backend::vexcl<double> Backend;

    typedef amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::spai0
        > AMG;

    amgcl::backend::crs<double, int> A;
    std::vector<double> rhs;

    prof.tic("assemble");
    int m = argc > 1 ? std::stoi(argv[1]) : 128;
    int n = A.nrows = A.ncols = sample_problem(m, A.val, A.col, A.ptr, rhs);
    prof.toc("assemble");

    prof.tic("build");
    AMG::params prm;
    prm.backend.q = ctx;

    AMG amg(A, prm);
    prof.toc("build");

    std::cout << amg << std::endl;

    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(ctx, n);
    x = 0;

    typedef amgcl::solver::bicgstab<Backend> Solver;
    Solver solve(n, Solver::params(), prm.backend);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(amg, f, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << prof << std::endl;
}
