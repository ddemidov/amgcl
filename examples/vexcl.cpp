#include <iostream>
#include <cstdlib>
#include <vexcl/vexcl.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_vexcl.hpp>
#include <amgcl/operations_vexcl.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    amgcl::profiler<> prof(argv[0]);

    // Read matrix and rhs from a binary file.
    std::vector<int>  row;
    std::vector<int>  col;
    std::vector<real> val;
    std::vector<real> rhs;
    int n = read_problem(argv[1], row, col, val, rhs);

    // Initialize VexCL context.
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );

    if (!ctx.size()) {
        std::cerr << "No GPUs" << std::endl;
        return 1;
    }

    std::cout << ctx << std::endl;

    // Wrap the matrix into amgcl::sparse::map:
    amgcl::sparse::matrix_map<real, int> A(
            n, n, row.data(), col.data(), val.data()
            );

    // Build the preconditioner.
    typedef amgcl::solver<
        real, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::vexcl<amgcl::relax::spai0>
        > AMG;

    typename AMG::params prm;
    // Provide vex::Context for level construction:
    prm.level.ctx = &ctx;
    // Use K-Cycle on each level to improve convergence:
    prm.level.kcycle = 1;

    prof.tic("setup");
    AMG amg(A, prm);
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Copy matrix and rhs to GPU(s).
    vex::SpMat<real, int, int> Agpu(
            ctx.queue(), n, n, row.data(), col.data(), val.data()
            );

    vex::vector<real> f(ctx.queue(), rhs);

    // Solve the problem with CG method. Use AMG as a preconditioner:
    vex::vector<real> x(ctx.queue(), n);
    x = 0;

    prof.tic("solve (cg)");
    auto cnv = amgcl::solve(Agpu, f, amg, x, amgcl::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    std::cout << prof;
}
