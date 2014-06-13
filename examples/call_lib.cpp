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

    amgclHandle amg = amgcl_create(
            amgclBackendBuiltin,
            amgclCoarseningSmoothedAggregation,
            amgclRelaxationSPAI0,
            n, ptr.data(), col.data(), val.data()
            );

    amgcl_solve(amgclSolverBiCGStab, amg, rhs.data(), x.data());

    amgcl_destroy(amg);
}
