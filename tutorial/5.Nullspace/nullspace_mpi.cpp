#include <vector>
#include <iostream>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/coarsening/rigid_body_modes.hpp>

#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/coarsening/smoothed_aggregation.hpp>
#include <amgcl/mpi/relaxation/spai0.hpp>
#include <amgcl/mpi/solver/cg.hpp>

#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

#if defined(AMGCL_HAVE_PARMETIS)
#  include <amgcl/mpi/partition/parmetis.hpp>
#elif defined(AMGCL_HAVE_SCOTCH)
#  include <amgcl/mpi/partition/ptscotch.hpp>
#endif

int main(int argc, char *argv[]) {
    // The command line should contain the matrix, the RHS, and the coordinate files:
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <A.bin> <b.bin> <coo.bin>" << std::endl;
        return 1;
    }

    amgcl::mpi::init mpi(&argc, &argv);
    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    // The profiler:
    amgcl::profiler<> prof("Nullspace");

    // Read the system matrix, the RHS, and the coordinates:
    prof.tic("read");
    // Get the global size of the matrix:
    ptrdiff_t rows = amgcl::io::crs_size<ptrdiff_t>(argv[1]);

    // Split the matrix into approximately equal chunks of rows, and
    // make sure each chunk size is divisible by 3.
    ptrdiff_t chunk = (rows + world.size - 1) / world.size;
    if (chunk % 3) chunk += 3 - chunk % 3;

    ptrdiff_t row_beg = std::min(rows, chunk * world.rank);
    ptrdiff_t row_end = std::min(rows, row_beg + chunk);
    chunk = row_end - row_beg;

    // Read our part of the system matrix, the RHS and the coordinates.
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs, coo;
    amgcl::io::read_crs(argv[1], rows, ptr, col, val, row_beg, row_end);

    ptrdiff_t n, m;
    amgcl::io::read_dense(argv[2], n, m, rhs, row_beg, row_end);
    amgcl::precondition(n == rows && m == 1, "The RHS file has wrong dimensions");

    amgcl::io::read_dense(argv[3], n, m, coo, row_beg / 3, row_end / 3);
    amgcl::precondition(n * 3 == rows && m == 3, "The coordinate file has wrong dimensions");
    prof.toc("read");

    if (world.rank == 0) {
        std::cout
            << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl
            << "RHS "    << argv[2] << ": " << rows << "x1" << std::endl
            << "Coords " << argv[3] << ": " << rows / 3 << "x3" << std::endl;
    }

    // Declare the backends and the solver type
    typedef amgcl::backend::builtin<double> SBackend; // the solver backend
    typedef amgcl::backend::builtin<float>  PBackend; // the preconditioner backend

    typedef amgcl::mpi::make_solver<
        amgcl::mpi::amg<
            PBackend,
            amgcl::mpi::coarsening::smoothed_aggregation<PBackend>,
            amgcl::mpi::relaxation::spai0<PBackend>
            >,
        amgcl::mpi::solver::cg<PBackend>
        > Solver;

    // The distributed matrix
    auto A = std::make_shared<amgcl::mpi::distributed_matrix<SBackend>>(
            world, std::tie(chunk, ptr, col, val));

    // Partition the matrix, the RHS vector, and the coordinates.
    // If neither ParMETIS not PT-SCOTCH are not available,
    // just keep the current naive partitioning.
#if defined(AMGCL_HAVE_PARMETIS) || defined(AMGCL_HAVE_SCOTCH)
#  if defined(AMGCL_HAVE_PARMETIS)
    typedef amgcl::mpi::partition::parmetis<SBackend> Partition;
#  elif defined(AMGCL_HAVE_SCOTCH)
    typedef amgcl::mpi::partition::ptscotch<SBackend> Partition;
#  endif

    if (world.size > 1) {
        auto t = prof.scoped_tic("partition");
        Partition part;

        // part(A) returns the distributed permutation matrix.
        // Keep the DOFs belonging to the same grid nodes together
        // (use block-wise partitioning with block size 3).
        auto P = part(*A, 3);
        auto R = transpose(*P);

        // Reorder the matrix:
        A = product(*R, *product(*A, *P));

        // Reorder the RHS vector and the coordinates:
        R->move_to_backend();
        std::vector<double> new_rhs(R->loc_rows());
        std::vector<double> new_coo(R->loc_rows());
        amgcl::backend::spmv(1, *R, rhs, 0, new_rhs);
        amgcl::backend::spmv(1, *R, coo, 0, new_coo);
        rhs.swap(new_rhs);
        coo.swap(new_coo);

        // Update the number of the local rows
        // (it may have changed as a result of permutation).
        chunk = A->loc_rows();
    }
#endif

    // Solver parameters:
    Solver::params prm;
    prm.solver.maxiter = 500;
    prm.precond.coarsening.aggr.eps_strong = 0;

    // Convert the coordinates to the rigid body modes.
    // The function returns the number of near null-space vectors
    // (3 in 2D case, 6 in 3D case) and writes the vectors to the
    // std::vector<double> specified as the last argument:
    prm.precond.coarsening.aggr.nullspace.cols = amgcl::coarsening::rigid_body_modes(
            3, coo, prm.precond.coarsening.aggr.nullspace.B);

    // Initialize the solver with the system matrix.
    prof.tic("setup");
    Solver solve(world, A, prm);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    if (world.rank == 0) std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(chunk, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = solve(*A, rhs, x);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    if (world.rank == 0) {
        std::cout
            << "Iters: " << iters << std::endl
            << "Error: " << error << std::endl
            << prof << std::endl;
    }
}
