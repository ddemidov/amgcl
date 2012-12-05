#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <vexcl/vexcl.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_vexcl.hpp>
#include <amgcl/operations_vexcl.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>

namespace amg {
amg::profiler<> prof;
}
using amg::prof;

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

    // Wrap the matrix into amg::sparse::map:
    amg::sparse::matrix_map<double, int> A(
            n, n, row.data(), col.data(), val.data()
            );

    // Initialize VexCL context.
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );

    if (!ctx.size()) {
        std::cerr << "No GPUs" << std::endl;
        return 1;
    }

    std::cout << ctx << std::endl;

    // Copy matrix and rhs to GPU(s).
    vex::SpMat<double, int, int> Agpu(
            ctx.queue(), n, n, row.data(), col.data(), val.data()
            );

    vex::vector<double> f(ctx.queue(), rhs);

    // Build the preconditioner.
    amg::params prm;
    prm.ncycle = 2;
    prm.over_interp = 1.5;

    prof.tic("setup");
    amg::solver<
        double, int,
        amg::interp::aggr_plain,
        amg::level::vexcl
        > amg(A, prm);
    prof.toc("setup");

    // Solve the problem with CG method. Use AMG as a preconditioner:
    vex::vector<double> x(ctx.queue(), n);
    x = 0;

    prof.tic("solve (cg)");
    auto cnv = amg::solve(Agpu, f, amg, x, amg::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    // Solve the problem with BiCGStab method. Use AMG as a preconditioner:
    x = 0;
    prof.tic("solve (bicg)");
    cnv = amg::solve(Agpu, f, amg, x, amg::bicg_tag());
    prof.toc("solve (bicg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    std::cout << prof;
}
