#ifndef AMGCL_PARAMS_HPP
#define AMGCL_PARAMS_HPP

namespace amg {

// Minimal set af AMG parameters.
struct params {
    float    eps_strong;    // Parameter for strong connections.
    unsigned coarse_enough; // When level is coarse enough to be solved directly.

    bool     trunc_int;     // Truncate prolongation operator.
    float    eps_tr;        // Truncation parameter.

    unsigned npre;          // Number of pre-relaxations.
    unsigned npost;         // Number of post-relaxations.
    unsigned ncycle;        // Number of cycles (1 for V-cycle, 2 for W-cycle, etc.).
    unsigned maxiter;       // Maximum number of iterations in standalone solver.
    double   tol;           // The required precision for standalone solver.

    params() {
        eps_strong    = 0.25f;
        coarse_enough = 300;
        trunc_int     = true;
        eps_tr        = 0.2f;
        npre          = 1;
        npost         = 1;
        ncycle        = 1;
        maxiter       = 100;
        tol           = 1e-8;
    }
};

} // namespace amg

#endif
