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
    amgclBackendBuiltin  = 1,
    amgclBackendBlockCRS = 2
} amgclBackend;

// Coarsening
typedef enum {
    amgclCoarseningRugeStuben          = 1,
    amgclCoarseningAggregation         = 2,
    amgclCoarseningSmoothedAggregation = 3,
    amgclCoarseningSmoothedAggrEMin    = 4
} amgclCoarsening;

// Relaxation
typedef enum {
    amgclRelaxationDampedJacobi = 1,
    amgclRelaxationGaussSeidel  = 2,
    amgclRelaxationChebyshev    = 3,
    amgclRelaxationSPAI0        = 4,
    amgclRelaxationILU0         = 5
} amgclRelaxation;

// Solver
typedef enum {
    amgclSolverCG        = 1,
    amgclSolverBiCGStab  = 2,
    amgclSolverBiCGStabL = 3,
    amgclSolverGMRES     = 4
} amgclSolver;

// Generic parameter list.
typedef void* amgclParams;

// Create parameter list.
amgclParams amgcl_params_create();

// Set integer parameter in a parameter list.
void amgcl_params_seti(amgclParams prm, const char *name, int   value);

// Set floating point parameter in a parameter list.
void amgcl_params_setf(amgclParams prm, const char *name, float value);

// Destroy parameter list.
void amgcl_params_destroy(amgclParams prm);

// AMG preconditioner data structure.
typedef void* amgclHandle;

// Create AMG preconditioner.
amgclHandle amgcl_create(
        amgclBackend    backend,
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        amgclParams     prm,
        size_t n,
        const long   *ptr,
        const long   *col,
        const double *val
        );

// Solve the problem for the given right-hand side.
void amgcl_solve(
        amgclSolver solver,
        amgclParams prm,
        amgclHandle amg,
        const double *rhs,
        double *x
        );

// Destroy AMG preconditioner.
void amgcl_destroy(amgclHandle handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
