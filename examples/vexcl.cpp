#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <amgcl/amgcl.hpp>
#include <amgcl/vexcl_level.hpp>
#include <vexcl/vexcl.hpp>
#include <vexcl/external/viennacl.hpp>
#include <viennacl/linalg/cg.hpp>

amg::profiler<> prof;

// Simple wrapper around amg::solver that provides ViennaCL's preconditioner
// interface.
struct amg_precond {
    // Build AMG hierarchy.
    template <class matrix>
    amg_precond(const matrix &A, const amg::params &prm = amg::params())
        : amg(A, prm), r(amg::sparse::matrix_rows(A))
    { }

    // Use one V-cycle with zero initial approximation as a preconditioning step.
    void apply(vex::vector<double> &x) const {
        r = 0;
        r.swap(x);
        amg.cycle(r, x);
    }

    // Build VexCL-based hierarchy:
    mutable amg::solver<
	double, int, amg::level::vexcl<double, int>
	> amg;
    mutable vex::vector<double> r;
};

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

    // Wrap the matrix into amg::sparse::amp:
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
    amg_precond amg(A);
    prof.toc("setup");

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, n);
    vex::vector<double> x = viennacl::linalg::solve(Agpu, f, tag, amg);
    prof.toc("solve");

    std::cout << "  Iterations: " << tag.iters() << std::endl
              << "  Error:      " << tag.error() << std::endl;

    std::cout << prof;
}
