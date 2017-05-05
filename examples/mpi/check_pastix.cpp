#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <boost/scope_exit.hpp>

#include <amgcl/mpi/pastix.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
    profiler<> prof;
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    const int n  = argc > 1 ? atoi(argv[1]) : 256;
    const int n2 = n * n;

    using amgcl::prof;

    int chunk       = (n2 + world.size - 1) / world.size;
    int chunk_start = world.rank * chunk;
    int chunk_end   = std::min(chunk_start + chunk, n2);

    chunk = chunk_end - chunk_start;

    std::vector<int> domain(world.size + 1);
    MPI_Allgather(&chunk, 1, MPI_INT, &domain[1], 1, MPI_INT, world);
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    prof.tic("assemble");
    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 5);
    val.reserve(chunk * 5);
    rhs.reserve(chunk);

    ptr.push_back(0);

    const double h2i  = (n - 1) * (n - 1);
    for(int idx = chunk_start; idx < chunk_end; ++idx) {
        int j = idx / n;
        int i = idx % n;

        if (j > 0)  {
            col.push_back(idx - n);
            val.push_back(-h2i);
        }

        if (i > 0) {
            col.push_back(idx - 1);
            val.push_back(-h2i);
        }

        col.push_back(idx);
        val.push_back(4 * h2i);

        if (i + 1 < n) {
            col.push_back(idx + 1);
            val.push_back(-h2i);
        }

        if (j + 1 < n) {
            col.push_back(idx + n);
            val.push_back(-h2i);
        }

        rhs.push_back(1);
        ptr.push_back( col.size() );
    }
    prof.toc("assemble");

    prof.tic("setup");
    amgcl::mpi::PaStiX<double> solve(world, chunk, ptr, col, val);
    prof.toc("setup");

    prof.tic("solve");
    std::vector<double> x(chunk);
    solve(rhs, x);
    solve(rhs, x);
    prof.toc("solve");

    prof.tic("save");
    if (world.rank == 0) {
        std::vector<double> X(n2);
        std::copy(x.begin(), x.end(), X.begin());

        for(int i = 1; i < world.size; ++i)
            MPI_Recv(&X[domain[i]], domain[i+1] - domain[i], MPI_DOUBLE, i, 42, world, MPI_STATUS_IGNORE);

        std::ofstream f("out.dat", std::ios::binary);
        f.write((char*)&n2, sizeof(int));
        f.write((char*)X.data(), n2 * sizeof(double));
    } else {
        MPI_Send(x.data(), chunk, MPI_DOUBLE, 0, 42, world);
    }
    prof.toc("save");

    if (world.rank == 0) {
        std::cout << prof << std::endl;
    }

}
