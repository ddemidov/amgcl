#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <vexcl/vexcl.hpp>

#define AMGCL_PROFILING
#define AGGREGATION

#include <amgcl/amgcl.hpp>
#ifdef AGGREGATION
#  include <amgcl/aggregation.hpp>
#else
#  include <amgcl/interp_classic.hpp>
#endif
#include <amgcl/level_vexcl.hpp>
#include <amgcl/operations_vexcl.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>

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
    std::ifstream pfile(argv[1], std::ios::binary);
    int n;
    pfile.read((char*)&n, sizeof(int));

    std::vector<int> row(n + 1);
    pfile.read((char*)row.data(), row.size() * sizeof(int));

    std::vector<int>    col(row.back());
    std::vector<double> val(row.back());
    std::vector<double> rhs(n);

    pfile.read((char*)col.data(), col.size() * sizeof(int));
    pfile.read((char*)val.data(), val.size() * sizeof(double));
    pfile.read((char*)rhs.data(), rhs.size() * sizeof(double));

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
    amgcl::params prm;
#ifdef AGGREGATION
    prm.kcycle = 1;
    prm.over_interp = 1.5;
#endif

    prof.tic("setup");
    amgcl::solver<
        double, int,
#ifdef AGGREGATION
        amgcl::interp::aggregation<amgcl::aggr::plain>,
#else
        amgcl::interp::classic,
#endif
        amgcl::level::vexcl
        > amg(A, prm);
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
