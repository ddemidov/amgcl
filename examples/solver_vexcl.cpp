#include <iostream>
#include <amgcl/backend/crs_tuple.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/backend/ccrs.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/coarsening/aggregation.hpp>
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

    typedef amgcl::backend::vexcl<double> backend;

    typedef amgcl::builder<
        double, amgcl::coarsening::aggregation
        > Builder;

    typedef amgcl::amg<
        backend, amgcl::relaxation::damped_jacobi
        > AMG;

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    prof.tic("assemble");
    int n = sample_problem(32, val, col, ptr, rhs);
    prof.toc("assemble");

    prof.tic("build");
    Builder builder( boost::tie(n, n, val, col, ptr) );
    prof.toc("build");

    std::cout << builder << std::endl;

    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;

    AMG::params prm;
    prm.backend.q = ctx.queue();

    prof.tic("move");
    AMG amg(builder, prm);
    prof.toc("move");

    vex::vector<double> x(ctx, n);
    vex::vector<double> f(ctx, rhs);

    amgcl::backend::clear(x);

    amgcl::solver::cg<backend> solve(n, prm.backend);
    size_t iters;
    double resid;

    prof.tic("solve");
    boost::tie(iters, resid) = solve(amg.top_matrix(), f, amg, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << amgcl::prof << std::endl;
}
