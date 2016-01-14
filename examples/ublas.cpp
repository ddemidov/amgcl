#include <iostream>
#include <cstdlib>
#include <utility>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <amgcl/amg.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/ublas.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

typedef boost::numeric::ublas::compressed_matrix<
    double, boost::numeric::ublas::row_major
    > ublas_matrix;

typedef boost::numeric::ublas::vector<double> ublas_vector;

namespace amgcl {
    profiler<> prof;
}

int main(int argc, char *argv[]) {
    using amgcl::prof;

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    prof.tic("assemble");
    int m = argc > 1 ? atoi(argv[1]) : 128;
    int n = sample_problem(m, val, col, ptr, rhs);

    // Create ublas matrix with the data.
    ublas_matrix A(n, n);
    A.reserve(ptr[n]);

    for(int i = 0; i < n; ++i)
        for(int j = ptr[i], e = ptr[i+1]; j < e; ++j)
            A.push_back(i, col[j], val[j]);
    prof.toc("assemble");

    prof.tic("build");
    amgcl::make_solver<
        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstabl<
            amgcl::backend::builtin<double>
            >
        > solve( amgcl::backend::map(A) );
    prof.toc("build");

    std::cout << solve.precond() << std::endl;

    ublas_vector x(n, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(rhs, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl << prof << std::endl;
}
