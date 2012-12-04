#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <amgcl/amgcl.hpp>
#include <amgcl/vexcl_level.hpp>
#include <amgcl/cg.hpp>
#include <vexcl/vexcl.hpp>

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
    prof.tic("setup");
    amg::solver<double, int, amg::level::vexcl<double, int>> amg(A);
    prof.toc("setup");

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    vex::vector<double> x(ctx.queue(), n);
    x = 0;
    amg::cg(Agpu, f, amg, x);
    prof.toc("solve");

    std::cout << prof;
}
