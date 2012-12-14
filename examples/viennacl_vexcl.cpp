#include <iostream>
#include <cstdlib>
#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_vexcl.hpp>
#include <vexcl/vexcl.hpp>
#include <vexcl/external/viennacl.hpp>
#include <viennacl/linalg/cg.hpp>

#include "read.hpp"

namespace amgcl {
profiler<> prof;
}
using amgcl::prof;

// Simple wrapper around amgcl::solver that provides ViennaCL's preconditioner
// interface.
struct amg_precond {
    typedef amgcl::solver<
        double, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::vexcl
        > AMG;
    typedef typename AMG::params params;

    // Build AMG hierarchy.
    template <class matrix>
    amg_precond(const matrix &A, const params &prm = params())
        : amg(A, prm), r(amgcl::sparse::matrix_rows(A))
    {
        std::cout << amg << std::endl;
    }

    // Use one V-cycle with zero initial approximation as a preconditioning step.
    void apply(vex::vector<double> &x) const {
        r = 0;
        r.swap(x);
        amg.apply(r, x);
    }

    // Build VexCL-based hierarchy:
    mutable AMG amg;
    mutable vex::vector<double> r;
};

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

    // Wrap the matrix into amgcl::sparse::map:
    amgcl::sparse::matrix_map<double, int> A(
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
    prof.tic("setup");
    amg_precond amg(A);
    prof.toc("setup");

    // Copy matrix and rhs to GPU(s).
    vex::SpMat<double, int, int> Agpu(
            ctx.queue(), n, n, row.data(), col.data(), val.data()
            );
    vex::vector<double> f(ctx.queue(), rhs);

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, n);
    vex::vector<double> x = viennacl::linalg::solve(Agpu, f, tag, amg);
    prof.toc("solve");

    std::cout << "Iterations: " << tag.iters() << std::endl
              << "Error:      " << tag.error() << std::endl;

    std::cout << prof;
}
