#include <vector>

#include <lib/amgcl.h>
#include "sample_problem.hpp"

int main() {
    std::vector<long>   ptr;
    std::vector<long>   col;
    std::vector<double> val;
    std::vector<double> rhs;

    size_t n = sample_problem(128l, val, col, ptr, rhs);

    std::vector<double> x(n, 0);

    amgclParams prm = amgcl_params_create();
    amgcl_params_seti(prm, "coarse_enough", 500);
    amgcl_params_setf(prm, "aggr.eps_strong", 1e-3);

    amgclHandle amg = amgcl_create(
            amgclBackendBuiltin,
            amgclCoarseningSmoothedAggregation,
            amgclRelaxationSPAI0,
            prm,
            n, ptr.data(), col.data(), val.data()
            );

    amgcl_params_seti(prm, "L", 1);
    amgcl_solve(amgclSolverBiCGStabL, prm, amg, rhs.data(), x.data());

    amgcl_destroy(amg);
    amgcl_params_destroy(prm);
}
