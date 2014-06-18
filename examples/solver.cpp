#include <iostream>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/crs_builder.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

struct poisson_2d {
    typedef double val_type;
    typedef long   col_type;

    size_t n;
    double h2i;

    poisson_2d(size_t n) : n(n), h2i((n - 1) * (n - 1)) {}

    size_t rows()     const { return n * n; }
    size_t nonzeros() const { return n * 5; }

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
    amgcl::profiler<> prof;


    int m = argc > 1 ? atoi(argv[1]) : 1024;
    int n = m * m;

    prof.tic("build");
    amgcl::make_solver<
        amgcl::backend::builtin<double>,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::spai0,
        amgcl::solver::bicgstab
        > solve( amgcl::backend::make_matrix(poisson_2d(m)) );
    prof.toc("build");

    std::cout << solve << std::endl;

    std::vector<double> f(n, 1);
    std::vector<double> x(n, 0);

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
