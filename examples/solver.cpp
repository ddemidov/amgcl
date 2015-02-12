#include <iostream>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_builder.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/profiler.hpp>

#include <boost/range/algorithm.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

//---------------------------------------------------------------------------
struct poisson_2d {
    typedef double    val_type;
    typedef ptrdiff_t col_type;

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

//---------------------------------------------------------------------------
template <class Vec>
double norm(const Vec &v) {
    return sqrt(amgcl::backend::inner_product(v, v));
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using amgcl::prof;

    int m = argc > 1 ? atoi(argv[1]) : 1024;
    int n = m * m;

    // Create iterative solver preconditioned by AMG.
    // The use of make_matrix() from crs_builder.hpp allows to construct the
    // system matrix on demand row by row.
    prof.tic("build");
    amgcl::make_solver<
        amgcl::backend::builtin<double>,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::spai0,
        amgcl::solver::cg
        > solve( amgcl::adapter::make_matrix( poisson_2d(m) ) );
    prof.toc("build");

    std::cout << solve.amg() << std::endl;

    std::vector<double> f(n, 1);
    std::vector<double> x(n, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(f, x);
    prof.toc("solve");

    std::cout << "Solver:" << std::endl
              << "  Iterations: " << iters << std::endl
              << "  Error:      " << resid << std::endl
              << std::endl;

    // Use the constructed solver as a preconditioner for another iterative
    // solver.
    //
    // Iterative methods use estimated residual for exit condition. For some
    // problems the value of estimated residual can get too far from true
    // residual due to round-off errors.
    //
    // Nesting iterative solvers in this way allows to shave last bits off the
    // error.
    amgcl::solver::cg< amgcl::backend::builtin<double> > S(n);
    boost::fill(x, 0);

    prof.tic("nested solver");
    boost::tie(iters, resid) = S(solve.amg().top_matrix(), solve, f, x);
    prof.toc("nested solver");

    std::cout << "Nested solver:" << std::endl
              << "  Iterations: " << iters << std::endl
              << "  Error:      " << resid << std::endl
              << std::endl;

    std::cout << prof << std::endl;
}
