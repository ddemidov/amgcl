#include <iostream>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <amgcl/runtime.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/direct_solver.hpp>
#include <amgcl/mpi/subdomain_deflation.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include "amgcl_mpi.h"

//---------------------------------------------------------------------------
typedef amgcl::backend::builtin<double>                   Backend;
typedef boost::property_tree::ptree                       Params;

typedef amgcl::mpi::make_solver<
    amgcl::mpi::subdomain_deflation<
        amgcl::runtime::amg<Backend>,
        amgcl::runtime::mpi::direct_solver<double>
        >,
    amgcl::runtime::iterative_solver
    > Solver;

//---------------------------------------------------------------------------
struct deflation_vectors {
    int n;
    amgclDefVecFunction user_func;
    void *user_data;

    deflation_vectors(int n, amgclDefVecFunction user_func, void *user_data)
        : n(n), user_func(user_func), user_data(user_data)
    {}

    int dim() const { return n; }

    double operator()(int i, ptrdiff_t j) const {
        return user_func(i, j, user_data);
    }
};

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_mpi_create(
        MPI_Comm             comm,
        ptrdiff_t            n,
        const ptrdiff_t     *ptr,
        const ptrdiff_t     *col,
        const double        *val,
        int                  n_def_vec,
        amgclDefVecFunction  def_vec_func,
        void                *def_vec_data,
        amgclHandle          params
        )
{
    boost::function<double(ptrdiff_t, unsigned)> dv = deflation_vectors(n_def_vec, def_vec_func, def_vec_data);
    boost::property_tree::ptree prm = *static_cast<Params*>(params);
    prm.put("precond.num_def_vec", n_def_vec);
    prm.put("precond.def_vec",     &dv);

    return static_cast<amgclHandle>(
            new Solver(
                comm,
                boost::make_tuple(
                    n,
                    boost::make_iterator_range(ptr, ptr + n + 1),
                    boost::make_iterator_range(col, col + ptr[n]),
                    boost::make_iterator_range(val, val + ptr[n])
                    ),
                prm
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

    size_t n = solver->size();

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
