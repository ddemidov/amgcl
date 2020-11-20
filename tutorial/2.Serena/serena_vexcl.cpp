#include <vector>
#include <iostream>

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

int main(int argc, char *argv[]) {
    // The matrix and the RHS file names should be in the command line options:
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx>" << std::endl;
        return 1;
    }

    // Create VexCL context. Set the environment variable OCL_DEVICE to
    // control which GPU to use in case multiple are available,
    // and use single device:
    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

    // Enable support for block-valued matrices in VexCL kernels:
    vex::scoped_program_header h1(ctx, amgcl::backend::vexcl_static_matrix_declaration<double,3>());
    vex::scoped_program_header h2(ctx, amgcl::backend::vexcl_static_matrix_declaration<float,3>());

    // The profiler:
    amgcl::profiler<> prof("Serena");

    // Read the system matrix:
    size_t rows, cols;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val;

    prof.tic("read");
    std::tie(rows, cols) = amgcl::io::mm_reader(argv[1])(ptr, col, val);
    std::cout << "Matrix " << argv[1] << ": " << rows << "x" << cols << std::endl;
    prof.toc("read");

    // The RHS is filled with ones:
    std::vector<double> f(rows, 1.0);

    // Scale the matrix so that it has the unit diagonal.
    // First, find the diagonal values:
    std::vector<double> D(rows, 1.0);
    for(size_t i = 0; i < rows; ++i) {
        for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
            if (col[j] == i) {
                D[i] = 1 / sqrt(val[j]);
                break;
            }
        }
    }

    // Then, apply the scaling in-place:
    for(size_t i = 0; i < rows; ++i) {
        for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
            val[j] *= D[i] * D[col[j]];
        }
        f[i] *= D[i];
    }

    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
    auto A = std::tie(rows, ptr, col, val);

    // Compose the solver type
    typedef amgcl::static_matrix<double, 3, 3> dmat_type; // matrix value type in double precision
    typedef amgcl::static_matrix<double, 3, 1> dvec_type; // the corresponding vector value type
    typedef amgcl::static_matrix<float,  3, 3> smat_type; // matrix value type in single precision

    typedef amgcl::backend::vexcl<dmat_type> SBackend; // the solver backend
    typedef amgcl::backend::vexcl<smat_type> PBackend; // the preconditioner backend
    
    typedef amgcl::make_solver<
        amgcl::amg<
            PBackend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::cg<SBackend>
        > Solver;

    // Solver parameters
    Solver::params prm;
    prm.solver.maxiter = 500;

    // Set the VexCL context in the backend parameters
    SBackend::params bprm;
    bprm.q = ctx;

    // Initialize the solver with the system matrix.
    // We use the block_matrix adapter to convert the matrix into the block
    // format on the fly:
    prof.tic("setup");
    auto Ab = amgcl::adapter::block_matrix<dmat_type>(A);
    Solver solve(Ab, prm, bprm);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(rows, 0.0);

    // Since we are using mixed precision, we have to transfer the system matrix to the GPU:
    prof.tic("GPU matrix");
    auto A_gpu = SBackend::copy_matrix(
            std::make_shared<amgcl::backend::crs<dmat_type>>(Ab), bprm);
    prof.toc("GPU matrix");

    // We reinterpret both the RHS and the solution vectors as block-valued,
    // and copy them to the VexCL vectors:
    auto f_ptr = reinterpret_cast<dvec_type*>(f.data());
    auto x_ptr = reinterpret_cast<dvec_type*>(x.data());
    vex::vector<dvec_type> F(ctx, rows / 3, f_ptr);
    vex::vector<dvec_type> X(ctx, rows / 3, x_ptr);

    prof.tic("solve");
    std::tie(iters, error) = solve(*A_gpu, F, X);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;
}
