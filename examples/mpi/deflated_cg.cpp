#ifndef SDD_CG_HPP
#define SDD_CG_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

#include <boost/range/algorithm.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/crs_tuple.hpp>
#include <amgcl/mpi/deflated_cg.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

int main(int argc, char *argv[]) {
    boost::mpi::environment  env;
    boost::mpi::communicator world;

    const long n  = argc > 1 ? atoi(argv[1]) : 1024;
    const long n2 = n * n;

    boost::array<long, 2> lo = { {0, 0} };
    boost::array<long, 2> hi = { {n - 1, n - 1} };

    amgcl::profiler<> prof;

    prof.tic("partition");
    domain_partition<2> part(lo, hi, world.size());
    const long chunk = part.size( world.rank() );

    std::vector<long> domain(world.size() + 1);
    all_gather(world, chunk, &domain[1]);
    boost::partial_sum(domain, domain.begin());

    const long chunk_start = domain[world.rank()];
    const long chunk_end   = domain[world.rank() + 1];

    std::vector<long> renum(n2);
    for(long j = 0, idx = 0; j < n; ++j) {
        for(long i = 0; i < n; ++i, ++idx) {
            boost::array<long, 2> p = {{i, j}};
            std::pair<int,long> v = part.index(p);
            renum[idx] = domain[v.first] + v.second;
        }
    }
    prof.toc("partition");

    prof.tic("assemble");
    std::vector<long>   ptr;
    std::vector<long>   col;
    std::vector<double> val;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 7);
    val.reserve(chunk * 7);

    ptr.push_back(0);

    const double h2i = (n - 1) * (n - 1);
    for(long j = 0, idx = 0, k = 0; j < n; ++j) {
        for(long i = 0; i < n; ++i, ++idx) {
            if (renum[idx] < chunk_start || renum[idx] >= chunk_end) continue;

            assert(renum[idx] - domain[world.rank()] == k);

            if (j > 0)  {
                col.push_back(renum[idx - n]);
                val.push_back(-h2i);
            }

            if (i > 0) {
                col.push_back(renum[idx - 1]);
                val.push_back(-h2i);
            }

            col.push_back(renum[idx]);
            val.push_back(4 * h2i);

            if (i + 1 < n) {
                col.push_back(renum[idx + 1]);
                val.push_back(-h2i);
            }

            if (j + 1 < n) {
                col.push_back(renum[idx + n]);
                val.push_back(-h2i);
            }

            ptr.push_back( col.size() );

            ++k;
        }
    }
    prof.toc("assemble");

    prof.tic("setup");
    amgcl::mpi::deflated_cg<double> solve(world, boost::tie(chunk, ptr, col, val) );
    prof.toc("setup");

    prof.tic("solve");
    std::vector<double> f(chunk, 1);
    std::vector<double> x(chunk, 0);
    solve(f, x);
    prof.toc("solve");

    prof.tic("save");
    if (world.rank() == 0) {
        std::vector<double> X(n2);
        boost::copy(x, X.begin());

        for(int i = 1; i < world.size(); ++i)
            world.recv(i, 42, &X[domain[i]], domain[i+1] - domain[i]);

        std::ofstream f("out.dat", std::ios::binary);
        int m = n2;
        f.write((char*)&m, sizeof(int));
        for(long i = 0; i < n2; ++i)
            f.write((char*)&X[renum[i]], sizeof(double));
    } else {
        world.send(0, 42, x.data(), chunk);
    }
    prof.toc("save");

    if (world.rank() == 0) std::cout << prof << std::endl;
}

#endif
