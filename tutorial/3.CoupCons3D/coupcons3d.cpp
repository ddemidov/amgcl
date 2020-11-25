#include <vector>
#include <iostream>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

int main(int argc, char *argv[]) {
    // The command line should contain the matrix file name:
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx>" << std::endl;
        return 1;
    }

    // The profiler:
    amgcl::profiler<> prof("Serena");

    // Read the system matrix:
    ptrdiff_t rows, cols;
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
    for(ptrdiff_t i = 0; i < rows; ++i) {
        for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
            if (col[j] == i) {
                D[i] = 1 / sqrt(val[j]);
                break;
            }
        }
    }

    // Then, apply the scaling in-place:
    for(ptrdiff_t i = 0; i < rows; ++i) {
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
    typedef amgcl::static_matrix<double, 4, 4> dmat_type; // matrix value type in double precision
    typedef amgcl::static_matrix<double, 4, 1> dvec_type; // the corresponding vector value type
    typedef amgcl::static_matrix<float,  4, 4> smat_type; // matrix value type in single precision

    typedef amgcl::backend::builtin<dmat_type> SBackend; // the solver backend
    typedef amgcl::backend::builtin<smat_type> PBackend; // the preconditioner backend

    typedef amgcl::make_solver<
        amgcl::amg<
            PBackend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::ilu0
            >,
        amgcl::solver::bicgstab<SBackend>
        > Solver;

    // Initialize the solver with the system matrix.
    // Use the block_matrix adapter to convert the matrix into
    // the block format on the fly:
    prof.tic("setup");
    auto Ab = amgcl::adapter::block_matrix<dmat_type>(A);
    Solver solve(Ab);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(rows, 0.0);

    // Reinterpret both the RHS and the solution vectors as block-valued:
    auto f_ptr = reinterpret_cast<dvec_type*>(f.data());
    auto x_ptr = reinterpret_cast<dvec_type*>(x.data());
    auto F = amgcl::make_iterator_range(f_ptr, f_ptr + rows / 4);
    auto X = amgcl::make_iterator_range(x_ptr, x_ptr + rows / 4);

    prof.tic("solve");
    std::tie(iters, error) = solve(Ab, F, X);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;
}
