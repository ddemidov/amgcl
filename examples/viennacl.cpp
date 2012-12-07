#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>

#define AMGCL_PROFILING

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_classic.hpp>
#include <amgcl/level_viennacl.hpp>

#include <viennacl/linalg/cg.hpp>

namespace amgcl {
profiler<> prof;
}
using amgcl::prof;

// Simple wrapper around amgcl::solver that provides ViennaCL's preconditioner
// interface.
struct amg_precond {
    // Build AMG hierarchy.
    template <class matrix>
    amg_precond(const matrix &A, const amgcl::params &prm = amgcl::params())
        : amg(A, prm), r(amgcl::sparse::matrix_rows(A))
    { }

    // Use one V-cycle with zero initial approximation as a preconditioning step.
    void apply(viennacl::vector<double> &x) const {
        r.clear();
        r.fast_swap(x);
        amg.apply(r, x);
    }

    // Build VexCL-based hierarchy:
    mutable amgcl::solver<
        double, int,
        amgcl::interp::classic,
        amgcl::level::ViennaCL
        > amg;

    mutable viennacl::vector<double> r;
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    try {
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

    // Wrap the matrix into amgcl::sparse::map:
    amgcl::sparse::matrix_map<double, int> A(
            n, n, row.data(), col.data(), val.data()
            );

    // Build the preconditioner.
    amgcl::params prm;
    //prm.kcycle = 1;
    //prm.over_interp = 1.5;
    prof.tic("setup");
    amg_precond amg(A, prm);
    prof.toc("setup");

    // Copy matrix and rhs to GPU(s).
    viennacl::compressed_matrix<double> Agpu;
    amgcl::copy(A, Agpu);

    viennacl::vector<double> f(n);
    viennacl::copy(rhs, f);

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, n);
    viennacl::vector<double> x = viennacl::linalg::solve(Agpu, f, tag, amg);
    prof.toc("solve");

    std::cout << "Iterations: " << tag.iters() << std::endl
              << "Error:      " << tag.error() << std::endl;

    std::cout << prof;
    } catch (const char *err) {
        std::cout << "error: " << err << std::endl;
    }
}
