#include <iostream>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/mkl.hpp>
#include <amgcl/adapter/crs_builder.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

struct poisson_2d {
    typedef double val_type;
    typedef long   col_type;

    size_t n;
    double h2i;

    poisson_2d(size_t n) : n(n), h2i((n - 1) * (n - 1)) {}

    size_t rows()     const { return n * n; }
    size_t nonzeros() const { return 5 * rows(); }

    void operator()(size_t row,
            std::vector<col_type> &col,
            std::vector<val_type> &val
            ) const
    {
        size_t i = row % n;
        size_t j = row / n;

        if (j > 0) {
            col.push_back(row - n);
            val.push_back(-h2i);
        }

        if (i > 0) {
            col.push_back(row - 1);
            val.push_back(-h2i);
        }

        col.push_back(row);
        val.push_back(4 * h2i);

        if (i + 1 < n) {
            col.push_back(row + 1);
            val.push_back(-h2i);
        }

        if (j + 1 < n) {
            col.push_back(row + n);
            val.push_back(-h2i);
        }
    }
};

int main(int argc, char *argv[]) {
    using amgcl::prof;

    int m = argc > 1 ? atoi(argv[1]) : 1024;
    int n = m * m;

    prof.tic("build");
    amgcl::make_solver<
        amgcl::backend::mkl,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::spai0,
        amgcl::solver::bicgstabl
        > solve( amgcl::adapter::make_matrix(poisson_2d(m)) );
    prof.toc("build");

    std::cout << solve.amg() << std::endl;

    amgcl::backend::mkl_vec f(n);
    amgcl::backend::mkl_vec x(n);

    std::fill_n(f.data(), n, 1.0);
    std::fill_n(x.data(), n, 0.0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(f, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << prof << std::endl;
}
