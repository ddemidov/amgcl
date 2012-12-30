#include <iostream>
#include <cstdlib>

#include <amgcl/amgcl.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/level_cuda.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }
    amgcl::profiler<> prof(argv[0]);

    // Read matrix and rhs from a binary file.
    std::vector<int>  row;
    std::vector<int>  col;
    std::vector<real> val;
    std::vector<real> rhs;
    int n = read_problem(argv[1], row, col, val, rhs);

    // Build the preconditioner:
    typedef amgcl::solver<
        real, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cuda<amgcl::sparse::CUDA_MATRIX_HYB, amgcl::relax::spai0>
        > AMG;

    AMG::params prm;
    prm.level.kcycle = 1;

    amgcl::sparse::matrix_map<real, int> A(
            n, n, row.data(), col.data(), val.data()
            );

    prof.tic("setup");
    AMG amg(A, prm);
    prof.toc("setup");

    std::cout << amg  << std::endl;

    thrust::device_vector<real> f(rhs.begin(), rhs.end());
    thrust::device_vector<real> x(n, 0);

    prof.tic("solve");
    std::pair<int, real> cnv = amgcl::solve(amg.top_matrix(), f, amg, x, amgcl::cg_tag());
    prof.toc("solve");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl
              << prof << std::endl;
}
