#include <vector>

#include <lib/amgcl.h>
#include "sample_problem.hpp"

int main() {
    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    int n = sample_problem(128l, val, col, ptr, rhs);

    std::vector<double> x(n, 0);

    amgclHandle prm = amgcl_params_create();
    amgcl_params_seti(prm, "coarse_enough", 1000);
    amgcl_params_setf(prm, "aggr.eps_strong", 1e-3);

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

    amgcl_solver_solve(solver, amg, rhs.data(), x.data());

    amgcl_solver_destroy(solver);
    amgcl_precond_destroy(amg);
    amgcl_params_destroy(prm);
}
