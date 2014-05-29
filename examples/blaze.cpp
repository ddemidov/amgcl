#include <iostream>
#include <cstdlib>
#include <blaze/Math.h>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_blaze.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;

namespace amgcl {
    profiler<> prof("blaze");
}
using amgcl::prof;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    // Read matrix and rhs from a binary file.
    std::vector<int>  row;
    std::vector<int>  col;
    std::vector<real> val;

    blaze::DynamicVector<real> rhs;
    int n = read_problem(argv[1], row, col, val, rhs);

    // Wrap the matrix into amgcl::sparse::map:
    amgcl::sparse::matrix_map<real, int> A(
            n, n, row.data(), col.data(), val.data()
            );

    // Build the preconditioner.
    typedef amgcl::solver<
        real, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::blaze<amgcl::relax::spai0>
        > AMG;

    AMG::params prm;

    prof.tic("setup");
    AMG amg(A, prm);
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Copy matrix to Blaze structure.
    blaze::CompressedMatrix<real> Ablaze(n, n);
    Ablaze.reserve(row.back());
    for(int i = 0; i < n; ++i) {
        for(int j = row[i]; j < row[i + 1]; ++j)
            Ablaze.append(i, col[j], val[j]);
        Ablaze.finalize(i);
    }

    // Solve the problem with CG method. Use AMG as a preconditioner:
    blaze::DynamicVector<real> x(n);
    x = 0;

    prof.tic("solve (cg)");
    auto cnv = amgcl::solve(Ablaze, rhs, amg, x, amgcl::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    std::cout << prof;
}
