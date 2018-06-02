#include <iostream>

#include <type_traits>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include "amgcl.h"

#ifdef AMGCL_PROFILING
#include <amgcl/profiler.hpp>
namespace amgcl {
    profiler<> prof;
}
#endif

//---------------------------------------------------------------------------
typedef amgcl::backend::builtin<double>           Backend;
typedef amgcl::amg<Backend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper> AMG;
typedef amgcl::runtime::solver::wrapper<Backend>  ISolver;
typedef amgcl::make_solver<AMG, ISolver>          Solver;
typedef boost::property_tree::ptree               Params;

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
void STDCALL amgcl_params_sets(amgclHandle prm, const char *name, const char *value) {
    static_cast<Params*>(prm)->put(name, value);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_read_json(amgclHandle prm, const char *fname) {
    read_json(fname, *static_cast<Params*>(prm));
}

//---------------------------------------------------------------------------
void STDCALL amgcl_params_destroy(amgclHandle prm) {
    delete static_cast<Params*>(prm);
}

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_precond_create(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        amgclHandle   prm
        )
{
    auto A = std::make_tuple(n,
            boost::make_iterator_range(ptr, ptr + n + 1),
            boost::make_iterator_range(col, col + ptr[n]),
            boost::make_iterator_range(val, val + ptr[n])
            );

    if (prm)
        return static_cast<amgclHandle>(new AMG(A, *static_cast<Params*>(prm)));
    else
        return static_cast<amgclHandle>(new AMG(A));
}

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_precond_create_f(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        amgclHandle   prm
        )
{
    auto ptr_c = boost::make_transform_iterator(ptr, [](int i){ return i - 1; });
    auto col_c = boost::make_transform_iterator(col, [](int i){ return i - 1; });

    auto A = std::make_tuple(n,
            boost::make_iterator_range(ptr_c, ptr_c + n + 1),
            boost::make_iterator_range(col_c, col_c + ptr[n]),
            boost::make_iterator_range(val, val + ptr[n])
            );

    if (prm)
        return static_cast<amgclHandle>(new AMG(A, *static_cast<Params*>(prm)));
    else
        return static_cast<amgclHandle>(new AMG(A));
}

//---------------------------------------------------------------------------
void STDCALL amgcl_precond_apply(amgclHandle handle, const double *rhs, double *x)
{
    AMG *amg = static_cast<AMG*>(handle);

    size_t n = amgcl::backend::rows(amg->system_matrix());

    boost::iterator_range<double*> x_range =
        boost::make_iterator_range(x, x + n);

    amg->apply(boost::make_iterator_range(rhs, rhs + n), x_range);
}

//---------------------------------------------------------------------------
void STDCALL amgcl_precond_report(amgclHandle handle) {
    std::cout << *static_cast<AMG*>(handle) << std::endl;
}

//---------------------------------------------------------------------------
void STDCALL amgcl_precond_destroy(amgclHandle handle) {
    delete static_cast<AMG*>(handle);
}

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_solver_create(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        amgclHandle   prm
        )
{
    auto A = std::make_tuple(n,
            boost::make_iterator_range(ptr, ptr + n + 1),
            boost::make_iterator_range(col, col + ptr[n]),
            boost::make_iterator_range(val, val + ptr[n])
            );

    if (prm)
        return static_cast<amgclHandle>(new Solver(A, *static_cast<Params*>(prm)));
    else
        return static_cast<amgclHandle>(new Solver(A));
}

//---------------------------------------------------------------------------
amgclHandle STDCALL amgcl_solver_create_f(
        int           n,
        const int    *ptr,
        const int    *col,
        const double *val,
        amgclHandle   prm
        )
{
    auto ptr_c = boost::make_transform_iterator(ptr, [](int i){ return i - 1; });
    auto col_c = boost::make_transform_iterator(col, [](int i){ return i - 1; });

    auto A = std::make_tuple(n,
            boost::make_iterator_range(ptr_c, ptr_c + n + 1),
            boost::make_iterator_range(col_c, col_c + ptr[n]),
            boost::make_iterator_range(val, val + ptr[n])
            );

    if (prm)
        return static_cast<amgclHandle>(new Solver(A, *static_cast<Params*>(prm)));
    else
        return static_cast<amgclHandle>(new Solver(A));
}

//---------------------------------------------------------------------------
void STDCALL amgcl_solver_report(amgclHandle handle) {
    std::cout << static_cast<Solver*>(handle)->precond() << std::endl;
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

    std::tie(cnv.iterations, cnv.residual) = (*slv)(
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

    std::tie(cnv.iterations, cnv.residual) = (*slv)(
            std::make_tuple(
                n,
                boost::make_iterator_range(A_ptr, A_ptr + n),
                boost::make_iterator_range(A_col, A_col + A_ptr[n]),
                boost::make_iterator_range(A_val, A_val + A_ptr[n])
                ),
            boost::make_iterator_range(rhs, rhs + n), x_range
            );

    return cnv;
}
