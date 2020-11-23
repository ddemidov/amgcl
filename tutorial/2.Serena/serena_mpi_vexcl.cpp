#include <vector>
#include <iostream>

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>

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

// Block size
const int B = 3;

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // The command line should contain the matrix file name:
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.bin>" << std::endl;
        return 1;
    }

    amgcl::mpi::init mpi(&argc, &argv);
    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    // Create VexCL context. Use vex::Filter::Exclusive so that different MPI
    // processes get different GPUs. Each process gets a single GPU:
    vex::Context ctx(vex::Filter::Exclusive(vex::Filter::Env && vex::Filter::Count(1)));
    for(int i = 0; i < world.size; ++i) {
        // unclutter the output:
        if (i == world.rank)
            std::cout << world.rank << ": " << ctx.queue(0) << std::endl;
        MPI_Barrier(world);
    }

    // Enable support for block-valued matrices in the VexCL kernels:
    vex::scoped_program_header h1(ctx, amgcl::backend::vexcl_static_matrix_declaration<double,B>());
    vex::scoped_program_header h2(ctx, amgcl::backend::vexcl_static_matrix_declaration<float,B>());

    // The profiler:
    amgcl::profiler<> prof("Serena MPI(VexCL)");

    prof.tic("read");
    // Get the global size of the matrix:
    size_t rows = amgcl::io::crs_size<size_t>(argv[1]);

    // Split the matrix into approximately equal chunks of rows, and
    // make sure each chunk size is divisible by the block size.
    size_t chunk = (rows + world.size - 1) / world.size;
    if (chunk % B) chunk += B - chunk % B;

    size_t row_beg = std::min(rows, chunk * world.rank);
    size_t row_end = std::min(rows, row_beg + chunk);
    chunk = row_end - row_beg;

    // Read our part of the system matrix.
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val;
    amgcl::io::read_crs(argv[1], rows, ptr, col, val, row_beg, row_end);
    prof.toc("read");

    if (world.rank == 0) std::cout
        << "World size: " << world.size << std::endl
        << "Matrix " << argv[1] << ": " << rows << "x" << rows << std::endl;

    // Declare the backend and the solver types
    typedef amgcl::static_matrix<double, B, B> dmat_type;
    typedef amgcl::static_matrix<double, B, 1> dvec_type;
    typedef amgcl::static_matrix<float,  B, B> fmat_type;
    typedef amgcl::backend::vexcl<dmat_type> DBackend;
    typedef amgcl::backend::vexcl<fmat_type> FBackend;

    typedef amgcl::mpi::make_solver<
        amgcl::mpi::amg<
            FBackend,
            amgcl::mpi::coarsening::smoothed_aggregation<FBackend>,
            amgcl::mpi::relaxation::spai0<FBackend>
            >,
        amgcl::mpi::solver::bicgstab<DBackend>
        > Solver;

    // Solver parameters
    Solver::params prm;
    prm.solver.maxiter = 200;

    // Set the VexCL context in the backend parameters
    DBackend::params bprm;
    bprm.q = ctx;

    // We need to scale the matrix, so that it has the unit diagonal.
    // Since we only have the local rows for the matrix, and we may need the
    // remote diagonal values, it is more convenient to represent the scaling
    // with the matrix-matrix product (As = D^-1/2 A D^-1/2).
    prof.tic("scale");
    // Find the local diagonal values,
    // and form the CRS arrays for a diagonal matrix.
    std::vector<double> dia(chunk, 1.0);
    std::vector<ptrdiff_t> d_ptr(chunk + 1), d_col(chunk);
    for(size_t i = 0, I = row_beg; i < chunk; ++i, ++I) {
        d_ptr[i] = i;
        d_col[i] = I;
        for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
            if (col[j] == I) {
                dia[i] = 1 / sqrt(val[j]);
                break;
            }
        }
    }
    d_ptr.back() = chunk;

    // Create the distributed diagonal matrix:
    amgcl::mpi::distributed_matrix<DBackend> D(world,
            amgcl::adapter::block_matrix<dmat_type>(
                std::tie(chunk, d_ptr, d_col, dia)));

    // The scaled matrix is formed as product D * A * D,
    // where A is the local chunk of the matrix
    // converted to the block format on the fly.
    auto A = product(D, *product(
                amgcl::mpi::distributed_matrix<DBackend>(world,
                    amgcl::adapter::block_matrix<dmat_type>(
                        std::tie(chunk, ptr, col, val))),
                D));
    prof.toc("scale");

    // Since the RHS in this case is filled with ones,
    // the scaled RHS is equal to dia.
    // Reinterpret the pointer to dia data to get the RHS in the block format:
    auto f_ptr = reinterpret_cast<dvec_type*>(dia.data());
    vex::vector<dvec_type> rhs(ctx, chunk / B, f_ptr);

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
        vex::vector<dvec_type> new_rhs(ctx, R->loc_rows());
        R->move_to_backend(bprm);
        amgcl::backend::spmv(1, *R, rhs, 0, new_rhs);
        rhs.swap(new_rhs);

        // Update the number of the local rows
        // (it may have changed as a result of permutation).
        // Note that A->loc_rows() returns the number of blocks,
        // as the matrix uses block values.
        chunk = A->loc_rows();
        prof.toc("partition");
    }
#endif

    // Initialize the solver:
    prof.tic("setup");
    Solver solve(world, A, prm, bprm);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    if (world.rank == 0) std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    vex::vector<dvec_type> x(ctx, chunk);
    x = amgcl::math::zero<dvec_type>();

    prof.tic("solve");
    std::tie(iters, error) = solve(*A, rhs, x);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    if (world.rank == 0) std::cout
        << "Iterations: " << iters << std::endl
        << "Error:      " << error << std::endl
        << prof << std::endl;
}
