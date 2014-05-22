#include <iostream>
#include <cstdlib>

#ifdef WIN32
#  undef AMGCL_PROFILING
#endif

#define BOOST_DISABLE_ASSERTS

#include <amgcl/amgcl.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/level_cuda.hpp>

#ifndef WIN32
// Boost.chrono used in profiler does not work with nvcc/windows combination.
#  include <amgcl/profiler.hpp>
#endif

#include "read.hpp"

typedef double real;

#ifndef WIN32
namespace amgcl {
    profiler<> prof("cuda");
}
using amgcl::prof;
#endif

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat> [dev_id]" << std::endl;
        return 1;
    }

    int dev_cnt;
    cudaGetDeviceCount(&dev_cnt);

    int dev_id;
    if (argc >= 3) {
        dev_id = atoi(argv[2]);
        if (dev_id < 0 || dev_id >= dev_cnt) {
            std::cerr << "Incorrect device id [0:" << dev_cnt << ")" << std::endl;
            return 1;
        }
        cudaSetDevice(dev_id);
    }

    {
        thrust::device_vector<int> dummy(1);
        cudaGetDevice(&dev_id);
    }

    for(int d = 0; d < dev_cnt; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        std::cout << d << (d == dev_id ? "* " : "  ") << p.name << std::endl;
    }
    std::cout << std::endl;

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
        amgcl::level::cuda<amgcl::GPU_MATRIX_HYB, amgcl::relax::spai0>
        > AMG;

    amgcl::sparse::matrix_map<real, int> A(
            n, n, row.data(), col.data(), val.data()
            );

#ifndef WIN32
    prof.tic("setup");
#endif
    AMG amg(A, AMG::params());
#ifndef WIN32
    prof.toc("setup");
#endif

    std::cout << amg  << std::endl;

    thrust::device_vector<real> f(rhs.begin(), rhs.end());
    thrust::device_vector<real> x(n, 0);

#ifndef WIN32
    prof.tic("solve");
#endif
    std::pair<int, real> cnv = amgcl::solve(amg.top_matrix(), f, amg, x,
            amgcl::cg_tag());
#ifndef WIN32
    prof.toc("solve");
#endif

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl;
#ifndef WIN32
    std::cout << std::endl << prof << std::endl;
#endif
}
