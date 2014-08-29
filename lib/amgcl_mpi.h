#ifndef LIB_AMGCL_MPI_H
#define LIB_AMGCL_MPI_H

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
 * \file   lib/amgcl_mpi.h
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  C wrapper interface to distributed amgcl solver.
 */

#include <mpi.h>
#include <amgcl.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double (STDCALL *amgclDefVecFunction)(int vec, long coo, void *data);

typedef enum {
    amgclDirectSolverSkylineLU
#ifdef AMGCL_HAVE_PASTIX
  , amgclDirectSolverPastix
#endif
} amgclDirectSolver;

// Create distributed solver.
amgclHandle STDCALL amgcl_mpi_create(
        amgclCoarsening      coarsening,
        amgclRelaxation      relaxation,
        amgclSolver          iterative_solver,
        amgclDirectSolver    direct_solver,
        amgclHandle          params,
        MPI_Comm             comm,
        long                 n,
        const long          *ptr,
        const long          *col,
        const double        *val,
        int                  n_def_vec,
        amgclDefVecFunction  def_vec_func,
        void                *def_vec_data
        );

// Find soltion for the given RHS.
conv_info STDCALL amgcl_mpi_solve(
        amgclHandle   solver,
        double const *rhs,
        double       *x
        );

// Destroy the distributed solver.
void STDCALL amgcl_mpi_destroy(amgclHandle solver);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
