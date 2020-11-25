#include <iostream>
#include <string>

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/make_block_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/preonly.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>

#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // The command line should contain the matrix and the RHS file names,
    // and the number of unknowns in the flow subsytem:
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix.bin> <rhs.bin> <nu>" << std::endl;
        return 1;
    }

    // Create VexCL context. Set the environment variable OCL_DEVICE to
    // control which GPU to use in case multiple are available,
    // and use single device:
    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

    // Enable support for block-valued matrices in the VexCL kernels:
    vex::scoped_program_header header(ctx, amgcl::backend::vexcl_static_matrix_declaration<float,3>());

    // The profiler:
    amgcl::profiler<> prof("UCube4 (VexCL)");

    // Read the system matrix:
    ptrdiff_t rows, cols;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;

    prof.tic("read");
    amgcl::io::read_crs(argv[1], rows, ptr, col, val);
    amgcl::io::read_dense(argv[2], rows, cols, rhs);
    std::cout << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl;
    std::cout << "RHS " << argv[2] << ": " << rows << "x" << cols << std::endl;
    prof.toc("read");

    // The number of unknowns in the U subsystem
    ptrdiff_t nu = std::stoi(argv[3]);

    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
    auto A = std::tie(rows, ptr, col, val);

    // Compose the solver type
    typedef amgcl::backend::vexcl<double> SBackend; // the outer iterative solver backend
    typedef amgcl::backend::vexcl<float>  PBackend; // the PSolver backend
    typedef amgcl::backend::vexcl<
        amgcl::static_matrix<float,3,3>> UBackend;    // the USolver backend

    typedef amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<
            amgcl::make_block_solver<
                amgcl::amg<
                    UBackend,
                    amgcl::coarsening::aggregation,
                    amgcl::relaxation::ilu0
                    >,
                amgcl::solver::preonly<UBackend>
                >,
            amgcl::make_solver<
                amgcl::relaxation::as_preconditioner<
                    PBackend,
                    amgcl::relaxation::spai0
                    >,
                amgcl::solver::preonly<PBackend>
                >
            >,
        amgcl::solver::cg<SBackend>
        > Solver;

    // Solver parameters
    Solver::params prm;
    prm.precond.simplec_dia = false;
    prm.precond.usolver.precond.relax.solve.iters = 4;
    prm.precond.pmask.resize(rows);
    for(ptrdiff_t i = 0; i < rows; ++i) prm.precond.pmask[i] = (i >= nu);

    // Set the VexCL context in the backend parameters
    SBackend::params bprm;
    bprm.q = ctx;

    // Initialize the solver with the system matrix.
    prof.tic("setup");
    Solver solve(A, prm, bprm);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // Since we are using mixed precision, we have to transfer the system matrix to the GPU:
    prof.tic("GPU matrix");
    auto A_gpu = SBackend::copy_matrix(std::make_shared<amgcl::backend::crs<double>>(A), bprm);
    prof.toc("GPU matrix");

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(ctx, rows);
    x = 0.0;

    prof.tic("solve");
    std::tie(iters, error) = solve(*A_gpu, f, x);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;
}
