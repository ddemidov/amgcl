#ifndef AMGCL_PARAMS_HPP
#define AMGCL_PARAMS_HPP

namespace amg {

// Minimal set af AMG parameters.
struct params {
    unsigned npre          = 1;	    // Number of pre-relaxations.
    unsigned npost         = 1;	    // Number of post-relaxations.
    unsigned ncycle        = 1;	    // Number of cycles (1 for V-cycle, 2 for W-cycle, etc.).
    unsigned maxiter       = 100;   // Maximum number of iterations in standalone solver.
    unsigned coarse_enough = 300;   // When level is coarse enough to be solved directly.

    float    eps_strong    = 0.25f; // Parameter for strong connections.
    double   tol           = 1e-8;  // The required precision for standalone solver.
};

} // namespace amg

#endif
