#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

#include <boost/scope_exit.hpp>
#include <boost/range/algorithm.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/mpi/subdomain_deflation.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

#define CONVECTION

namespace amgcl {
    profiler<> prof;
}

struct linear_deflation {
    std::vector<double> x;
    std::vector<double> y;

    linear_deflation(int n) : x(n), y(n) {}

    size_t dim() const { return 3; }

    double operator()(int i, int j) const {
        switch(j) {
            default:
            case 0:
                return 1;
            case 1:
                return x[i];
            case 2:
                return y[i];
        }
    }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    const int n  = argc > 1 ? atoi(argv[1]) : 1024;
    const int n2 = n * n;

    boost::array<int, 2> lo = { {0, 0} };
    boost::array<int, 2> hi = { {n - 1, n - 1} };

    using amgcl::prof;

    prof.tic("partition");
    domain_partition<2> part(lo, hi, world.size);
    int chunk = part.size( world.rank );

    std::vector<int> domain(world.size + 1);
    MPI_Allgather(&chunk, 1, MPI_LONG, &domain[1], 1, MPI_LONG, world);
    boost::partial_sum(domain, domain.begin());

    int chunk_start = domain[world.rank];
    int chunk_end   = domain[world.rank + 1];

    linear_deflation lindef(chunk);
    std::vector<int> renum(n2);
    for(int j = 0, idx = 0; j < n; ++j) {
        for(int i = 0; i < n; ++i, ++idx) {
            boost::array<int, 2> p = {{i, j}};
            std::pair<int,int> v = part.index(p);
            renum[idx] = domain[v.first] + v.second;

            boost::array<int,2> lo = part.domain(v.first).min_corner();
            boost::array<int,2> hi = part.domain(v.first).max_corner();

            if (v.first == world.rank) {
                lindef.x[v.second] = (i - (lo[0] + hi[0]) / 2);
                lindef.y[v.second] = (j - (lo[1] + hi[1]) / 2);
            }
        }
    }
    prof.toc("partition");

    prof.tic("assemble");
    std::vector<int>   ptr;
    std::vector<int>   col;
    std::vector<double> val;
    std::vector<double> rhs;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 5);
    val.reserve(chunk * 5);
    rhs.reserve(chunk);

    ptr.push_back(0);

    const double hinv = (n - 1);
    const double h2i  = (n - 1) * (n - 1);

    for(int j = 0, idx = 0; j < n; ++j) {
        for(int i = 0; i < n; ++i, ++idx) {
            if (renum[idx] < chunk_start || renum[idx] >= chunk_end) continue;

            if (j > 0)  {
                col.push_back(renum[idx - n]);
                val.push_back(-h2i);
            }

            if (i > 0) {
                col.push_back(renum[idx - 1]);
                val.push_back(-h2i - hinv);
            }

            col.push_back(renum[idx]);
            val.push_back(4 * h2i + hinv);

            if (i + 1 < n) {
                col.push_back(renum[idx + 1]);
                val.push_back(-h2i);
            }

            if (j + 1 < n) {
                col.push_back(renum[idx + n]);
                val.push_back(-h2i);
            }

            rhs.push_back(1);
            ptr.push_back( col.size() );
        }
    }
    prof.toc("assemble");

    prof.tic("setup");
    typedef amgcl::mpi::subdomain_deflation<
        amgcl::backend::vexcl<double>,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::spai0,
        amgcl::solver::bicgstabl
        > Solver;

    vex::Context ctx( vex::Filter::Exclusive(
                vex::Filter::Env &&
                //vex::Filter::DoublePrecision &&
                vex::Filter::Count(1)
                ) );

    std::cout << world.rank << ": " << ctx.queue(0) << std::endl;

    Solver::params prm;
    prm.amg.backend.q = ctx;

    Solver solve(world, boost::tie(chunk, ptr, col, val), lindef, prm);
    prof.toc("setup");

    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(ctx, chunk);
    x = 0;
    ctx.finish();

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(f, x);
    prof.toc("solve");

    if (n < 4096) {
        prof.tic("save");
        if (world.rank == 0) {
            std::vector<double> X(n2);
            vex::copy(x.begin(), x.end(), X.begin());

            for(int i = 1; i < world.size; ++i)
                MPI_Recv(&X[domain[i]], domain[i+1] - domain[i], MPI_DOUBLE, i, 42, world, MPI_STATUS_IGNORE);

            std::ofstream f("out.dat", std::ios::binary);
            int m = n2;
            f.write((char*)&m, sizeof(int));
            for(int i = 0; i < n2; ++i)
                f.write((char*)&X[renum[i]], sizeof(double));
        } else {
            std::vector<double> X(chunk);
            vex::copy(x, X);
            MPI_Send(X.data(), chunk, MPI_DOUBLE, 0, 42, world);
        }
        prof.toc("save");
    }

    if (world.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << resid << std::endl
            << std::endl
            << prof << std::endl;
    }
}
