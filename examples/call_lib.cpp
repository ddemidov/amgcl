#include <vector>

#include <boost/range/algorithm.hpp>
#include <lib/amgcl.h>
#include "sample_problem.hpp"

int main() {
    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    int n = sample_problem(128l, val, col, ptr, rhs);

    amgclHandle prm = amgcl_params_create();
    amgcl_params_seti(prm, "coarse_enough", 1000);
    amgcl_params_setf(prm, "coarsening.aggr.eps_strong", 1e-3);

    amgclHandle amg = amgcl_precond_create(
            amgclBackendBuiltin,
            amgclCoarseningSmoothedAggregation,
            amgclRelaxationSPAI0,
            prm,
            n, ptr.data(), col.data(), val.data()
            );

    amgcl_params_seti(prm, "L", 1);
    amgclHandle solver = amgcl_solver_create(
            amgclBackendBuiltin,
            amgclSolverBiCGStabL,
            prm, n
            );

    std::vector<double> x(n, 0);
    amgcl_solver_solve(solver, amg, rhs.data(), x.data());

    // Solve same problem again, but explicitly provide the matrix this time:
    boost::fill(x, 0);
    amgcl_solver_solve_mtx(solver, ptr.data(), col.data(), val.data(), amg,
            rhs.data(), x.data());

    amgcl_solver_destroy(solver);
    amgcl_precond_destroy(amg);
    amgcl_params_destroy(prm);
}
