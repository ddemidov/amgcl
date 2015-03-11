#include <iostream>

#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <amgcl/runtime.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include "amgcl.h"

#ifdef AMGCL_PROFILING
#include <amgcl/profiler.hpp>
namespace amgcl {
    profiler<> prof;
}
#endif

#define ASSERT_EQUAL(e1, e2) BOOST_STATIC_ASSERT((int)(e1) == (int)(e2))

ASSERT_EQUAL(amgclCoarseningRugeStuben,          amgcl::runtime::coarsening::ruge_stuben);
ASSERT_EQUAL(amgclCoarseningAggregation,         amgcl::runtime::coarsening::aggregation);
ASSERT_EQUAL(amgclCoarseningSmoothedAggregation, amgcl::runtime::coarsening::smoothed_aggregation);
ASSERT_EQUAL(amgclCoarseningSmoothedAggrEMin,    amgcl::runtime::coarsening::smoothed_aggr_emin);

ASSERT_EQUAL(amgclRelaxationGaussSeidel,         amgcl::runtime::relaxation::gauss_seidel);
ASSERT_EQUAL(amgclRelaxationMCGaussSeidel,       amgcl::runtime::relaxation::multicolor_gauss_seidel);
ASSERT_EQUAL(amgclRelaxationILU0,                amgcl::runtime::relaxation::ilu0);
ASSERT_EQUAL(amgclRelaxationDampedJacobi,        amgcl::runtime::relaxation::damped_jacobi);
ASSERT_EQUAL(amgclRelaxationSPAI0,               amgcl::runtime::relaxation::spai0);
ASSERT_EQUAL(amgclRelaxationSPAI1,               amgcl::runtime::relaxation::spai1);
ASSERT_EQUAL(amgclRelaxationChebyshev,           amgcl::runtime::relaxation::chebyshev);

ASSERT_EQUAL(amgclSolverCG,                      amgcl::runtime::solver::cg);
ASSERT_EQUAL(amgclSolverBiCGStab,                amgcl::runtime::solver::bicgstab);
ASSERT_EQUAL(amgclSolverBiCGStabL,               amgcl::runtime::solver::bicgstabl);
ASSERT_EQUAL(amgclSolverGMRES,                   amgcl::runtime::solver::gmres);

//---------------------------------------------------------------------------
typedef amgcl::backend::builtin<double>      Backend;
typedef amgcl::runtime::amg<Backend>         AMG;
typedef amgcl::runtime::make_solver<Backend> Solver;
typedef boost::property_tree::ptree          Params;

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_params_create() {
    return static_cast<amgclHandle>( new Params() );
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_seti(amgclHandle prm, const char *name, int value) {
    static_cast<Params*>(prm)->put(name, value);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_setf(amgclHandle prm, const char *name, float value) {
    static_cast<Params*>(prm)->put(name, value);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_destroy(amgclHandle prm) {
    delete static_cast<Params*>(prm);
}

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_precond_create(
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        amgclHandle     prm,
        int n,
        const int    *ptr,
        const int    *col,
        const double *val
        )
{
    return static_cast<amgclHandle>(
            new AMG(
                static_cast<amgcl::runtime::coarsening::type>(coarsening),
                static_cast<amgcl::runtime::relaxation::type>(relaxation),
                boost::make_tuple(
                    n,
                    boost::make_iterator_range(ptr, ptr + n + 1),
                    boost::make_iterator_range(col, col + ptr[n]),
                    boost::make_iterator_range(val, val + ptr[n])
                    ),
                *static_cast<Params*>(prm)
                )
            );
}

//---------------------------------------------------------------------------
void STDCALL amgcl_precond_apply(amgclHandle handle, const double *rhs, double *x)
{
    AMG *amg = static_cast<AMG*>(handle);

    size_t n = amg->size();

    boost::iterator_range<double*> x_range =
        boost::make_iterator_range(x, x + n);

    amg->apply(boost::make_iterator_range(rhs, rhs + n), x_range);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_precond_destroy(amgclHandle handle) {
    delete static_cast<AMG*>(handle);
}

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_solver_create(
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        amgclSolver     solver,
        amgclHandle     prm,
        int n,
        const int    *ptr,
        const int    *col,
        const double *val
        )
{
    return static_cast<amgclHandle>(
            new Solver(
                static_cast<amgcl::runtime::coarsening::type>(coarsening),
                static_cast<amgcl::runtime::relaxation::type>(relaxation),
                static_cast<amgcl::runtime::solver::type>(solver),
                boost::make_tuple(
                    n,
                    boost::make_iterator_range(ptr, ptr + n + 1),
                    boost::make_iterator_range(col, col + ptr[n]),
                    boost::make_iterator_range(val, val + ptr[n])
                    ),
                *static_cast<Params*>(prm)
                )
            );
}

//---------------------------------------------------------------------------
void STDCALL amgcl_solver_destroy(amgclHandle handle) {
    delete static_cast<Solver*>(handle);
}

//---------------------------------------------------------------------------
conv_info STDCALL amgcl_solver_solve(
        amgclHandle handle,
        const double *rhs,
        double *x
        )
{
    Solver *slv = static_cast<Solver*>(handle);

    size_t n = slv->size();

    conv_info cnv;

    boost::iterator_range<double*> x_range = boost::make_iterator_range(x, x + n);

    boost::tie(cnv.iterations, cnv.residual) = (*slv)(
            boost::make_iterator_range(rhs, rhs + n), x_range
            );

    return cnv;
}

//---------------------------------------------------------------------------
conv_info STDCALL amgcl_solver_solve_mtx(
        amgclHandle handle,
        int    const * A_ptr,
        int    const * A_col,
        double const * A_val,
        const double *rhs,
        double *x
        )
{
    Solver *slv = static_cast<Solver*>(handle);

    size_t n = slv->size();

    conv_info cnv;

    boost::iterator_range<double*> x_range = boost::make_iterator_range(x, x + n);

    boost::tie(cnv.iterations, cnv.residual) = (*slv)(
            boost::make_tuple(
                n,
                boost::make_iterator_range(A_ptr, A_ptr + n),
                boost::make_iterator_range(A_col, A_col + A_ptr[n]),
                boost::make_iterator_range(A_val, A_val + A_ptr[n])
                ),
            boost::make_iterator_range(rhs, rhs + n), x_range
            );

    return cnv;
}
