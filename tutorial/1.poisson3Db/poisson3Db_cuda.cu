#include <vector>
#include <iostream>

#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

int main(int argc, char *argv[]) {
    // The matrix and the RHS file names should be in the command line options:
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <rhs.mtx>" << std::endl;
        return 1;
    }

    // Show the name of the GPU we are using:
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << prop.name << std::endl;

    // The profiler:
    amgcl::profiler<> prof("poisson3Db");

    // Read the system matrix and the RHS:
    size_t rows, cols;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;

    prof.tic("read");
    std::tie(rows, cols) = amgcl::io::mm_reader(argv[1])(ptr, col, val);
    std::cout << "Matrix " << argv[1] << ": " << rows << "x" << cols << std::endl;

    std::tie(rows, cols) = amgcl::io::mm_reader(argv[2])(rhs);
    std::cout << "RHS " << argv[2] << ": " << rows << "x" << cols << std::endl;
    prof.toc("read");

    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
    auto A = std::tie(rows, ptr, col, val);

    // Compose the solver type
    typedef amgcl::backend::cuda<double> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<Backend>
        > Solver;

    // We need to initialize the CUSPARSE library and pass the handle to AMGCL
    // in backend parameters:
    Backend::params bprm;
    cusparseCreate(&bprm.cusparse_handle);

    // There is no way to pass the backend parameters without passing the
    // solver parameters, so we also need to create those. But we can leave
    // them with the default values:
    Solver::params prm;

    // Initialize the solver with the system matrix:
    prof.tic("setup");
    Solver solve(A, prm, bprm);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation.
    // The RHS and the solution vectors should reside in the GPU memory:
    int iters;
    double error;
    thrust::device_vector<double> f(rhs);
    thrust::device_vector<double> x(rows, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = solve(f, x);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;
}
