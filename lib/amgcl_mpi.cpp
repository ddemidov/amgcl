#include <iostream>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <amgcl/runtime.hpp>
#include <amgcl/mpi/runtime.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include "amgcl_mpi.h"

#define ASSERT_EQUAL(e1, e2) BOOST_STATIC_ASSERT((int)(e1) == (int)(e2))

ASSERT_EQUAL(amgclDirectSolverSkylineLU, amgcl::runtime::direct_solver::skyline_lu);
#ifdef AMGCL_HAVE_PASTIX
ASSERT_EQUAL(amgclDirectSolverPastix,    amgcl::runtime::direct_solver::pastix);
#endif

//---------------------------------------------------------------------------
typedef amgcl::backend::builtin<double>                   Backend;
typedef amgcl::runtime::mpi::subdomain_deflation<Backend> Solver;
typedef boost::property_tree::ptree                       Params;

//---------------------------------------------------------------------------
struct deflation_vectors {
    int n;
    amgclDefVecFunction user_func;
    void *user_data;

    deflation_vectors(int n, amgclDefVecFunction user_func, void *user_data)
        : n(n), user_func(user_func), user_data(user_data)
    {}

    int dim() const { return n; }

    double operator()(int i, long j) const {
        return user_func(i, j, user_data);
    }
};

//---------------------------------------------------------------------------
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
        )
{
    return static_cast<amgclHandle>(
            new Solver(
                static_cast<amgcl::runtime::coarsening::type>(coarsening),
                static_cast<amgcl::runtime::relaxation::type>(relaxation),
                static_cast<amgcl::runtime::solver::type>(iterative_solver),
                static_cast<amgcl::runtime::direct_solver::type>(direct_solver),
                comm,
                boost::make_tuple(
                    n,
                    boost::make_iterator_range(ptr, ptr + n + 1),
                    boost::make_iterator_range(col, col + ptr[n]),
                    boost::make_iterator_range(val, val + ptr[n])
                    ),
                deflation_vectors(n_def_vec, def_vec_func, def_vec_data),
                *static_cast<Params*>(params)
                )
            );
}

//---------------------------------------------------------------------------
conv_info STDCALL amgcl_mpi_solve(
        amgclHandle   handle,
        double const *rhs,
        double       *x
        )
{
    Solver *solver = static_cast<Solver*>(handle);

    size_t n = solver->local_size();

    boost::iterator_range<double*> x_range =
        boost::make_iterator_range(x, x + n);

    conv_info cnv;

    boost::tie(cnv.iterations, cnv.residual) = (*solver)(
            boost::make_iterator_range(rhs, rhs + n), x_range
            );

    return cnv;
}

//---------------------------------------------------------------------------
void STDCALL amgcl_mpi_destroy(amgclHandle handle) {
    delete static_cast<Solver*>(handle);
}
