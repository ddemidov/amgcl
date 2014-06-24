#ifndef SDD_CG_HPP
#define SDD_CG_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

#include <boost/range/algorithm.hpp>
#include <boost/scope_exit.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/mpi/deflation.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

#define CONVECTION

struct linear_deflation {
    long n;
    double h;
    std::vector<long> idx;

    linear_deflation(long n) : n(n), h(1.0 / (n - 1)), idx(n * n) {}

    size_t dim() const { return 3; }

    double operator()(long i, int j) const {
        switch(j) {
            case 1:
                return h * (idx[i] % n);
            case 2:
                return h * (idx[i] / n);
            case 0:
            default:
                return 1;
        }
    }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    const long n  = argc > 1 ? atoi(argv[1]) : 1024;
    const long n2 = n * n;

    boost::array<long, 2> lo = { {0, 0} };
    boost::array<long, 2> hi = { {n - 1, n - 1} };

    amgcl::profiler<> prof;

    prof.tic("partition");
    domain_partition<2> part(lo, hi, world.size);
    const long chunk = part.size( world.rank );

    std::vector<long> domain(world.size + 1);
    MPI_Allgather(&chunk, 1, MPI_LONG, &domain[1], 1, MPI_LONG, world);
    boost::partial_sum(domain, domain.begin());

    const long chunk_start = domain[world.rank];
    const long chunk_end   = domain[world.rank + 1];

    linear_deflation lindef(n);
    std::vector<long> renum(n2);
    for(long j = 0, idx = 0; j < n; ++j) {
        for(long i = 0; i < n; ++i, ++idx) {
            boost::array<long, 2> p = {{i, j}};
            std::pair<int,long> v = part.index(p);
            renum[idx] = domain[v.first] + v.second;
            lindef.idx[renum[idx]] = idx;
        }
    }
    prof.toc("partition");

    prof.tic("assemble");
    std::vector<long>   ptr;
    std::vector<long>   col;
    std::vector<double> val;
    std::vector<double> rhs;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 5);
    val.reserve(chunk * 5);
    rhs.reserve(chunk);

    ptr.push_back(0);

    const double hinv = (n - 1);
    const double h2i  = (n - 1) * (n - 1);
    for(long j = 0, idx = 0; j < n; ++j) {
        for(long i = 0; i < n; ++i, ++idx) {
            if (renum[idx] < chunk_start || renum[idx] >= chunk_end) continue;

            if (j > 0)  {
                col.push_back(renum[idx - n]);
                val.push_back(-h2i);
            }

            if (i > 0) {
                col.push_back(renum[idx - 1]);
                val.push_back(-h2i
#ifdef CONVECTION
                        - hinv
#endif
                        );
            }

            col.push_back(renum[idx]);
            val.push_back(4 * h2i
#ifdef CONVECTION
                    + hinv
#endif
                    );

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
                vex::Filter::DoublePrecision &&
                vex::Filter::Count(1)
                ) );

    typename Solver::AMG_params    amg_prm;
    amg_prm.backend.q = ctx;

    typename Solver::Solver_params slv_prm(2, 500, 1e-6);

    Solver solve(world,
            boost::tie(chunk, ptr, col, val),
            lindef,
            amg_prm, slv_prm
            );
    prof.toc("setup");

    prof.tic("solve");
    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(ctx, chunk);
    x = 0;

    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(f, x);
    prof.toc("solve");

    prof.tic("save");
    if (world.rank == 0) {
        std::vector<double> X(n2);
        vex::copy(x.begin(), x.end(), X.begin());

        for(int i = 1; i < world.size; ++i)
            MPI_Recv(&X[domain[i]], domain[i+1] - domain[i], MPI_DOUBLE, i, 42, world, MPI_STATUS_IGNORE);

        std::ofstream f("out.dat", std::ios::binary);
        int m = n2;
        f.write((char*)&m, sizeof(int));
        for(long i = 0; i < n2; ++i)
            f.write((char*)&X[renum[i]], sizeof(double));
    } else {
        std::vector<double> X(chunk);
        vex::copy(x, X);
        MPI_Send(X.data(), chunk, MPI_DOUBLE, 0, 42, world);
    }
    prof.toc("save");

    if (world.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << resid << std::endl
            << std::endl
            << prof << std::endl;
    }
}

#endif
