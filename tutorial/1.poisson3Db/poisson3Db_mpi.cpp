#include <vector>
#include <iostream>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/coarsening/smoothed_aggregation.hpp>
#include <amgcl/mpi/relaxation/spai0.hpp>
#include <amgcl/mpi/solver/bicgstab.hpp>

#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

#if defined(AMGCL_HAVE_PARMETIS)
#  include <amgcl/mpi/partition/parmetis.hpp>
#elif defined(AMGCL_HAVE_SCOTCH)
#  include <amgcl/mpi/partition/ptscotch.hpp>
#endif

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // The matrix and the RHS file names should be in the command line options:
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix.bin> <rhs.bin>" << std::endl;
        return 1;
    }

    amgcl::mpi::init mpi(&argc, &argv);
    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    // The profiler:
    amgcl::profiler<> prof("poisson3Db MPI");

    // Read the system matrix and the RHS:
    prof.tic("read");
    // Get the global size of the matrix:
    ptrdiff_t rows = amgcl::io::crs_size<ptrdiff_t>(argv[1]);
    ptrdiff_t cols;

    // Split the matrix into approximately equal chunks of rows
    ptrdiff_t chunk = (rows + world.size - 1) / world.size;
    ptrdiff_t row_beg = std::min(rows, chunk * world.rank);
    ptrdiff_t row_end = std::min(rows, row_beg + chunk);
    chunk = row_end - row_beg;

    // Read our part of the system matrix and the RHS.
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;
    amgcl::io::read_crs(argv[1], rows, ptr, col, val, row_beg, row_end);
    amgcl::io::read_dense(argv[2], rows, cols, rhs, row_beg, row_end);
    prof.toc("read");

    if (world.rank == 0)
        std::cout
            << "World size: " << world.size << std::endl
            << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl
            << "RHS " << argv[2] << ": " << rows << "x" << cols << std::endl;

    // Compose the solver type
    typedef amgcl::backend::builtin<double> DBackend;
    typedef amgcl::backend::builtin<float>  FBackend;
    typedef amgcl::mpi::make_solver<
        amgcl::mpi::amg<
            FBackend,
            amgcl::mpi::coarsening::smoothed_aggregation<FBackend>,
            amgcl::mpi::relaxation::spai0<FBackend>
            >,
        amgcl::mpi::solver::bicgstab<DBackend>
        > Solver;

    // Create the distributed matrix from the local parts.
    auto A = std::make_shared<amgcl::mpi::distributed_matrix<DBackend>>(
            world, std::tie(chunk, ptr, col, val));

    // Partition the matrix and the RHS vector.
    // If neither ParMETIS not PT-SCOTCH are not available,
    // just keep the current naive partitioning.
#if defined(AMGCL_HAVE_PARMETIS) || defined(AMGCL_HAVE_SCOTCH)
#  if defined(AMGCL_HAVE_PARMETIS)
    typedef amgcl::mpi::partition::parmetis<DBackend> Partition;
#  elif defined(AMGCL_HAVE_SCOTCH)
    typedef amgcl::mpi::partition::ptscotch<DBackend> Partition;
#  endif

    if (world.size > 1) {
        prof.tic("partition");
        Partition part;

        // part(A) returns the distributed permutation matrix:
        auto P = part(*A);
        auto R = transpose(*P);

        // Reorder the matrix:
        A = product(*R, *product(*A, *P));

        // and the RHS vector:
        std::vector<double> new_rhs(R->loc_rows());
        R->move_to_backend(typename DBackend::params());
        amgcl::backend::spmv(1, *R, rhs, 0, new_rhs);
        rhs.swap(new_rhs);

        // Update the number of the local rows
        // (it may have changed as a result of permutation):
        chunk = A->loc_rows();
        prof.toc("partition");
    }
#endif

    // Initialize the solver:
    prof.tic("setup");
    Solver solve(world, A);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    if (world.rank == 0)
        std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(chunk, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = solve(*A, rhs, x);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    if (world.rank == 0)
        std::cout
            << "Iters: " << iters << std::endl
            << "Error: " << error << std::endl
            << prof << std::endl;
}
