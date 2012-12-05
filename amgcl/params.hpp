#ifndef AMGCL_PARAMS_HPP
#define AMGCL_PARAMS_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

namespace amg {

// Minimal set af AMG parameters.
struct params {
    float    eps_strong;    // Parameter for strong connections.
    unsigned coarse_enough; // When level is coarse enough to be solved directly.

    bool     trunc_int;     // Truncate prolongation operator.
    float    eps_tr;        // Truncation parameter.
    float    over_interp;   // Over-interpolation factor.

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
        over_interp   = 1.0f;
        npre          = 1;
        npost         = 1;
        ncycle        = 1;
        maxiter       = 100;
        tol           = 1e-8;
    }
};

} // namespace amg

#endif
