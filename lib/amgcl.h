#ifndef LIB_AMGCL_H
#define LIB_AMGCL_H

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

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

/**
 * \file   lib/amgcl.h
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  C wrapper interface to amgcl.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Backends
typedef enum {
    amgclBackendBuiltin  = 0,
    amgclBackendBlockCRS = 1
} amgclBackend;

// Coarsening
typedef enum {
    amgclCoarseningRugeStuben          = 0,
    amgclCoarseningAggregation         = 1,
    amgclCoarseningSmoothedAggregation = 2,
    amgclCoarseningSmoothedAggrEMin    = 3
} amgclCoarsening;

// Relaxation
typedef enum {
    amgclRelaxationDampedJacobi = 0,
    amgclRelaxationSPAI0        = 1,
    amgclRelaxationChebyshev    = 2
} amgclRelaxation;

// Solver
typedef enum {
    amgclSolverCG       = 0,
    amgclSolverBiCGStab = 1,
    amgclSolverGMRES    = 2
} amgclSolver;

typedef void* amgclHandle;

amgclHandle amgcl_create(
        amgclBackend    backend,
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        size_t n,
        const long   *ptr,
        const long   *col,
        const double *val
        );

void amgcl_solve(
        amgclSolver solver,
        amgclHandle amg,
        const double *rhs,
        double *x
        );

void amgcl_destroy(amgclHandle handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
