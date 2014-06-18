#ifndef SDD_CG_HPP
#define SDD_CG_HPP

#include <fstream>
#include <vector>
#include <cmath>

#include <boost/range/algorithm.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/crs_tuple.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>

#include <amgcl/mpi/deflated_cg.hpp>

void add_node(long n, long idx, long i, long j,
        std::vector<long>   &ptr,
        std::vector<long>   &col,
        std::vector<double> &val,
        std::vector<double> &rhs
        )
{
    const double h2i = (n - 1) * (n - 1);

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

    ptr.push_back( static_cast<long>(col.size()) );
}

int main(int argc, char *argv[]) {
    boost::mpi::environment  env;
    boost::mpi::communicator world;

    std::vector<long>   ptr;
    std::vector<long>   col;
    std::vector<double> val;
    std::vector<double> rhs;

    const long n  = 1024;
    const long n2 = n * n;
    const long chunk_size = (n2 + world.size() - 1) / world.size();
    const long chunk_start = chunk_size * world.rank();
    const long chunk_end   = std::min(chunk_start + chunk_size, n2);
    const long chunk = chunk_end - chunk_start;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 7);
    val.reserve(chunk * 7);
    rhs.reserve(chunk);

    ptr.push_back(0);

    for(long j = 0, idx = 0; j < n; ++j) {
        for(long i = 0; i < n; ++i, ++idx) {
            if (idx >= chunk_start && idx < chunk_end)
                add_node(n, idx, i, j, ptr, col, val, rhs);
        }
    }

    std::vector<double> x(chunk, 0);

    amgcl::mpi::deflated_cg<double> solve(world, boost::tie(chunk, ptr, col, val) );
    solve(rhs, x);

    std::vector<double> X;

    if (world.rank() == 0) X.resize(n2);

    gather(world, x.data(), chunk, X.data(), 0);

    if (world.rank() == 0) {
        int m = n2;

        std::ofstream f("out.dat", std::ios::binary);
        f.write((char*)&m, sizeof(int));
        f.write((char*)X.data(), n2 * sizeof(double));
    }
}

#endif
