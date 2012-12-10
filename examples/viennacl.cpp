#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>

#define VIENNACL_WITH_OPENCL
//#define VIENNACL_WITH_OPENMP

#define AMGCL_PROFILING

#include <amgcl/amgcl.hpp>
#include <amgcl/level_viennacl.hpp>

#ifdef VIENNACL_WITH_OPENCL
#  include <vexcl/devlist.hpp>
#  include <viennacl/hyb_matrix.hpp>
#  include <amgcl/aggregation.hpp>
#else
#  include <viennacl/compressed_matrix.hpp>
#  include <amgcl/interp_classic.hpp>
#endif

#include <viennacl/vector.hpp>
#include <viennacl/linalg/cg.hpp>

namespace amgcl {
profiler<> prof;
}
using amgcl::prof;

// Simple wrapper around amgcl::solver that provides ViennaCL's preconditioner
// interface.
struct amg_precond {
    typedef amgcl::solver<
        double, int,
#ifdef VIENNACL_WITH_OPENCL
        amgcl::interp::aggregation<amgcl::aggr::plain>,
        amgcl::level::ViennaCL<amgcl::level::CL_MATRIX_HYB>
#else
        amgcl::interp::classic,
        amgcl::level::ViennaCL<amgcl::level::CL_MATRIX_CRS>
#endif
        > AMG;
    typedef typename AMG::params params;

    // Build AMG hierarchy.
    template <class matrix>
    amg_precond(const matrix &A, const params &prm = params())
        : amg(A, prm), r(amgcl::sparse::matrix_rows(A))
    { }

    // Use one V-cycle with zero initial approximation as a preconditioning step.
    void apply(viennacl::vector<double> &x) const {
        r.clear();
        amg.apply(x, r);
        viennacl::copy(r, x);
    }

    // Build VexCL-based hierarchy:
    mutable AMG amg;
    mutable viennacl::vector<double> r;
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

#ifdef VIENNACL_WITH_OPENCL
    // There is no easy way to select compute device in ViennaCL, so just use
    // VexCL for that.
    vex::Context ctx(
            vex::Filter::Env &&
            vex::Filter::DoublePrecision &&
            vex::Filter::Count(1)
            );
    std::vector<cl_device_id> dev_id(1, ctx.queue(0).getInfo<CL_QUEUE_DEVICE>()());
    std::vector<cl_command_queue> queue_id(1, ctx.queue(0)());
    viennacl::ocl::setup_context(0, ctx.context(0)(), dev_id, queue_id);
    std::cout << ctx << std::endl;
#endif

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
    amg_precond::params prm;
#ifdef VIENNACL_WITH_OPENCL
    prm.level.kcycle = 1;
#endif

    prof.tic("setup");
    amg_precond amg(A, prm);
    prof.toc("setup");

    // Copy matrix and rhs to GPU(s).
    viennacl::hyb_matrix<double> Agpu;
    viennacl::copy(amgcl::sparse::viennacl_map(A), Agpu);

    viennacl::vector<double> f(n);
    viennacl::fast_copy(rhs, f);

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, 100);
    viennacl::vector<double> x = viennacl::linalg::solve(Agpu, f, tag, amg);
    prof.toc("solve");

    std::cout << "Iterations: " << tag.iters() << std::endl
              << "Error:      " << tag.error() << std::endl;

    std::cout << prof;

    // Prevent ViennaCL from segfaulting:
    exit(0);
}
