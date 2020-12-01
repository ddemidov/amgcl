#include <vector>
#include <iostream>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/rigid_body_modes.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

int main(int argc, char *argv[]) {
    // The command line should contain the matrix, the RHS, and the coordinate files:
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <A.mtx> <b.mtx> <coo.mtx>" << std::endl;
        return 1;
    }

    // The profiler:
    amgcl::profiler<> prof("Nullspace");

    // Read the system matrix, the RHS, and the coordinates:
    ptrdiff_t rows, cols, ndim, ncoo;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs, coo;

    prof.tic("read");
    std::tie(rows, rows) = amgcl::io::mm_reader(argv[1])(ptr, col, val);
    std::tie(rows, cols) = amgcl::io::mm_reader(argv[2])(rhs);
    std::tie(ncoo, ndim) = amgcl::io::mm_reader(argv[3])(coo);
    prof.toc("read");

    amgcl::precondition(ncoo * ndim == rows && (ndim == 2 || ndim == 3),
            "The coordinate file has wrong dimensions");

    std::cout << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl;
    std::cout << "RHS "    << argv[2] << ": " << rows << "x" << cols << std::endl;
    std::cout << "Coords " << argv[3] << ": " << ncoo << "x" << ndim << std::endl;

    // Declare the solver type
    typedef amgcl::backend::builtin<double> SBackend; // the solver backend
    typedef amgcl::backend::builtin<float>  PBackend; // the preconditioner backend

    typedef amgcl::make_solver<
        amgcl::amg<
            PBackend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::cg<SBackend>
        > Solver;

    // Solver parameters:
    Solver::params prm;
    prm.precond.coarsening.aggr.eps_strong = 0;

    // Convert the coordinates to the rigid body modes.
    // The function returns the number of near null-space vectors
    // (3 in 2D case, 6 in 3D case) and writes the vectors to the
    // std::vector<double> specified as the last argument:
    prm.precond.coarsening.nullspace.cols = amgcl::coarsening::rigid_body_modes(
            ndim, coo, prm.precond.coarsening.nullspace.B);

    // We use the tuple of CRS arrays to represent the system matrix.
    auto A = std::tie(rows, ptr, col, val);

    // Initialize the solver with the system matrix.
    prof.tic("setup");
    Solver solve(A, prm);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(rows, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = solve(A, rhs, x);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;
}
