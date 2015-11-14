#include <iostream>
#include <thrust/device_vector.h>

#include <amgcl/amgcl.hpp>
#include <amgcl/make_solver.hpp>

#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/cusparse_ilu0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof("cuda");
}

int main(int argc, char *argv[]) {
    const int m = argc > 1 ? atoi(argv[1]) : 64;

    std::vector<int>    ptr, col;
    std::vector<double> val, rhs;

    using amgcl::prof;

    // 3d poisson in m*m*m cube:
    prof.tic("assemble");
    int n = sample_problem(m, val, col, ptr, rhs);
    prof.toc("assemble");

    // Setup solver:
    typedef amgcl::backend::cuda<double> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::ilu0
            >,
        amgcl::solver::bicgstab< Backend >
        > Solver;

    // Init CUSPARSE (once per program lifespan):
    Solver::params  sprm;
    Backend::params bprm;
    cusparseCreate(&bprm.cusparse_handle);

    prof.tic("setup");
    Solver solve( boost::tie(n, ptr, col, val), sprm, bprm );
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    // Solve the problem. The rhs and the solution vectors are in GPU memory.
    thrust::device_vector<double> f = rhs;
    thrust::device_vector<double> x(n);
    thrust::fill(x.begin(), x.end(), 0.0); // Initial approximation.

    int    iters;
    double error;

    prof.tic("solve");
    boost::tie(iters, error) = solve(f, x);
    prof.toc("solve");

    std::cout
        << "Iterations: " << iters << std::endl
        << "Error:      " << error << std::endl
        ;

    std::cout << prof << std::endl;
}

