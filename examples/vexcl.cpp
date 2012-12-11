#include <iostream>
#include <cstdlib>
#include <vexcl/vexcl.hpp>

#define AMGCL_PROFILING
#define AGGREGATION

#include <amgcl/amgcl.hpp>
#ifdef AGGREGATION
#  include <amgcl/aggr_plain.hpp>
#  include <amgcl/interp_aggr.hpp>
#else
#  include <amgcl/interp_classic.hpp>
#endif
#include <amgcl/level_vexcl.hpp>
#include <amgcl/operations_vexcl.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>

#include "read.hpp"

namespace amgcl {
profiler<> prof;
}
using amgcl::prof;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    // Read matrix and rhs from a binary file.
    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;
    int n = read_problem(argv[1], row, col, val, rhs);

    // Initialize VexCL context.
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );

    if (!ctx.size()) {
        std::cerr << "No GPUs" << std::endl;
        return 1;
    }

    std::cout << ctx << std::endl;

    // Wrap the matrix into amgcl::sparse::map:
    amgcl::sparse::matrix_map<double, int> A(
            n, n, row.data(), col.data(), val.data()
            );

    // Build the preconditioner.
    typedef amgcl::solver<
        double, int,
#ifdef AGGREGATION
        amgcl::interp::aggregation<amgcl::aggr::plain>,
#else
        amgcl::interp::classic,
#endif
        amgcl::level::vexcl
        > AMG;

    typename AMG::params prm;
    prm.level.ctx = &ctx;
#ifdef AGGREGATION
    prm.level.kcycle = 1;
#endif

    prof.tic("setup");
    AMG amg(A, prm);
    prof.toc("setup");

    // Copy matrix and rhs to GPU(s).
    vex::SpMat<double, int, int> Agpu(
            ctx.queue(), n, n, row.data(), col.data(), val.data()
            );

    vex::vector<double> f(ctx.queue(), rhs);

    // Solve the problem with CG method. Use AMG as a preconditioner:
    vex::vector<double> x(ctx.queue(), n);
    x = 0;

    prof.tic("solve (cg)");
    auto cnv = amgcl::solve(Agpu, f, amg, x, amgcl::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    // Solve the problem with BiCGStab method. Use AMG as a preconditioner:
    x = 0;
    prof.tic("solve (bicg)");
    cnv = amgcl::solve(Agpu, f, amg, x, amgcl::bicg_tag());
    prof.toc("solve (bicg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    std::cout << prof;
}
