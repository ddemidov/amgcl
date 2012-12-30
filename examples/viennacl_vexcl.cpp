#include <iostream>
#include <cstdlib>

#include <vexcl/vexcl.hpp>
#include <vexcl/external/viennacl.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_vexcl.hpp>
#include <amgcl/operations_viennacl.hpp>
#include <amgcl/profiler.hpp>

#include <viennacl/linalg/cg.hpp>

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

    // Wrap the matrix into amgcl::sparse::map:
    amgcl::sparse::matrix_map<real, int> A(
            n, n, row.data(), col.data(), val.data()
            );

    // Initialize VexCL context.
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    if (!ctx.size()) {
        std::cerr << "No GPUs" << std::endl;
        return 1;
    }
    std::cout << ctx << std::endl;

    // Build the preconditioner.
    typedef amgcl::solver<
        real, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::vexcl<amgcl::relax::damped_jacobi>
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

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, n);
    vex::vector<real> x = viennacl::linalg::solve(Agpu, f, tag,
            amgcl::make_viennacl_precond<vex::vector<real>>(amg));
    prof.toc("solve");

    std::cout << "Iterations: " << tag.iters() << std::endl
              << "Error:      " << tag.error() << std::endl;

    std::cout << prof;
}
