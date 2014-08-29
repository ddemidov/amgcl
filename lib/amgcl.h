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

#ifdef WIN32
#  define STDCALL __stdcall
#else
#  define STDCALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Coarsening
typedef enum {
    amgclCoarseningRugeStuben,
    amgclCoarseningAggregation,
    amgclCoarseningSmoothedAggregation,
    amgclCoarseningSmoothedAggrEMin
} amgclCoarsening;

// Relaxation
typedef enum {
    amgclRelaxationGaussSeidel,
    amgclRelaxationILU0,
    amgclRelaxationDampedJacobi,
    amgclRelaxationSPAI0,
    amgclRelaxationChebyshev
} amgclRelaxation;

// Solver
typedef enum {
    amgclSolverCG,
    amgclSolverBiCGStab,
    amgclSolverBiCGStabL,
    amgclSolverGMRES
} amgclSolver;

typedef void* amgclHandle;

// Create parameter list.
amgclHandle STDCALL amgcl_params_create();

// Set integer parameter in a parameter list.
void STDCALL amgcl_params_seti(amgclHandle prm, const char *name, int   value);

// Set floating point parameter in a parameter list.
void STDCALL amgcl_params_setf(amgclHandle prm, const char *name, float value);

// Destroy parameter list.
void STDCALL amgcl_params_destroy(amgclHandle prm);

// Create AMG preconditioner.
amgclHandle STDCALL amgcl_precond_create(
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        amgclHandle     parameters,
        int n,
        const int    *ptr,
        const int    *col,
        const double *val
        );

// Apply AMG preconditioner (x = M^(-1) * rhs).
void STDCALL amgcl_precond_apply(amgclHandle amg, const double *rhs, double *x);

// Destroy AMG preconditioner
void STDCALL amgcl_precond_destroy(amgclHandle amg);

// Create iterative solver preconditioned by AMG.
amgclHandle STDCALL amgcl_solver_create(
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        amgclSolver     solver,
        amgclHandle     parameters,
        int n,
        const int    *ptr,
        const int    *col,
        const double *val
        );

// Convergence info
struct conv_info {
    int    iterations;
    double residual;
};

// Solve the problem for the given right-hand side.
conv_info STDCALL amgcl_solver_solve(
        amgclHandle    solver,
        double const * rhs,
        double       * x
        );

// Solve the problem for the given matrix and the right-hand side.
conv_info STDCALL amgcl_solver_solve_mtx(
        amgclHandle    solver,
        int    const * A_ptr,
        int    const * A_col,
        double const * A_val,
        double const * rhs,
        double       * x
        );

// Destroy iterative solver.
void STDCALL amgcl_solver_destroy(amgclHandle solver);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
