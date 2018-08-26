#include <vector>
#include <tuple>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl { profiler<> prof; }
using amgcl::prof;

int main() {
    // Combine single-precision preconditioner with a
    // double-precision Krylov solver.
    typedef
        amgcl::amg<
            amgcl::backend::builtin<float>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >
        Precond;

    // Solver is in double precision:
    typedef
        amgcl::solver::cg<
            amgcl::backend::builtin<double>
            >
        Solver;

    std::vector<int>    ptr, col;
    std::vector<double> val_d;
    std::vector<double> rhs;

    prof.tic("assemble");
    int n = sample_problem(128, val_d, col, ptr, rhs);
    prof.toc("assemble");

    std::vector<double> x(n, 0.0);
    std::vector<float>  val_f(val_d.begin(), val_d.end());

    auto A_f = std::tie(n, ptr, col, val_f);
    auto A_d = std::tie(n, ptr, col, val_d);

    prof.tic("setup");
    Solver S(n);
    Precond P(A_f);
    prof.toc("setup");

    std::cout << P << std::endl;

    int iters;
    double error;
    prof.tic("solve");
    std::tie(iters, error) = S(A_d, P, rhs, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << prof << std::endl;
}
