#include <iostream>

#include <boost/type_traits.hpp>

#include <amgcl/amgcl.hpp>

#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/backend/builtin.hpp>

#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/pointwise_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggr_emin.hpp>

#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/chebyshev.hpp>

#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/gmres.hpp>

#include <amgcl/mpi/subdomain_deflation.hpp>
#include <amgcl/mpi/skyline_lu.hpp>
#ifdef HAVE_PASTIX
#include <amgcl/mpi/pastix.hpp>
#endif

#include "amgcl_mpi.h"

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    template <class> class Relaxation,
    template <class, class> class Solver,
    class Func
    >
static void process_sdd(
        amgclDirectSolver direct_solver,
        const Func &func
        )
{
    switch (direct_solver) {
        case amgclDirectSolverSkylineLU:
            func.template process<
                amgcl::mpi::subdomain_deflation<
                    Backend,
                    Coarsening,
                    Relaxation,
                    Solver,
                    amgcl::mpi::skyline_lu<double>
                    >
                >();
            break;
#ifdef HAVE_PASTIX
        case amgclDirectSolverPastix:
            func.template process<
                amgcl::mpi::subdomain_deflation<
                    Backend,
                    Coarsening,
                    Relaxation,
                    Solver,
                    amgcl::mpi::PaStiX<double>
                    >
                >();
            break;
#endif
    }
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    template <class> class Relaxation,
    class Func
    >
static void process_sdd(
        amgclSolver       solver,
        amgclDirectSolver direct_solver,
        const Func &func
        )
{
    switch (solver) {
        case amgclSolverCG:
            process_sdd<
                Backend,
                Coarsening,
                Relaxation,
                amgcl::solver::cg
                >(direct_solver, func);
            break;
        case amgclSolverBiCGStab:
            process_sdd<
                Backend,
                Coarsening,
                Relaxation,
                amgcl::solver::bicgstab
                >(direct_solver, func);
            break;
        case amgclSolverBiCGStabL:
            process_sdd<
                Backend,
                Coarsening,
                Relaxation,
                amgcl::solver::bicgstabl
                >(direct_solver, func);
            break;
        case amgclSolverGMRES:
            process_sdd<
                Backend,
                Coarsening,
                Relaxation,
                amgcl::solver::gmres
                >(direct_solver, func);
            break;
    }
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    class Func
    >
static typename boost::enable_if<
    boost::is_same<
        amgcl::backend::builtin<typename Backend::value_type>,
        Backend
        >,
    void
    >::type
process_sdd(
        amgclRelaxation   relaxation,
        amgclSolver       solver,
        amgclDirectSolver direct_solver,
        const Func &func
        )
{
    switch (relaxation) {
        case amgclRelaxationDampedJacobi:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::damped_jacobi
                >(solver, direct_solver, func);
            break;
        case amgclRelaxationGaussSeidel:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::gauss_seidel
                >(solver, direct_solver, func);
            break;
        case amgclRelaxationChebyshev:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::chebyshev
                >(solver, direct_solver, func);
            break;
        case amgclRelaxationSPAI0:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::spai0
                >(solver, direct_solver, func);
            break;
        case amgclRelaxationILU0:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::ilu0
                >(solver, direct_solver, func);
            break;
    }
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    class Func
    >
static typename boost::disable_if<
    boost::is_same<
        amgcl::backend::builtin<typename Backend::value_type>,
        Backend
        >,
    void
    >::type
process_sdd(
        amgclRelaxation   relaxation,
        amgclSolver       solver,
        amgclDirectSolver direct_solver,
        const Func &func
        )
{
    switch (relaxation) {
        case amgclRelaxationDampedJacobi:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::damped_jacobi
                >(solver, direct_solver, func);
            break;
        case amgclRelaxationChebyshev:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::chebyshev
                >(solver, direct_solver, func);
            break;
        case amgclRelaxationSPAI0:
            process_sdd<
                Backend,
                Coarsening,
                amgcl::relaxation::spai0
                >(solver, direct_solver, func);
            break;
        default:
            amgcl::precondition(false, "Unsupported relaxation scheme");
    }
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Func
    >
static void process_sdd(
        amgclCoarsening   coarsening,
        amgclRelaxation   relaxation,
        amgclSolver       solver,
        amgclDirectSolver direct_solver,
        const Func &func
        )
{
    switch (coarsening) {
        case amgclCoarseningRugeStuben:
            process_sdd<
                Backend,
                amgcl::coarsening::ruge_stuben
                >(relaxation, solver, direct_solver, func);
            break;
        case amgclCoarseningAggregation:
            process_sdd<
                Backend,
                amgcl::coarsening::aggregation<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, solver, direct_solver, func);
            break;
        case amgclCoarseningSmoothedAggregation:
            process_sdd<
                Backend,
                amgcl::coarsening::smoothed_aggregation<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, solver, direct_solver, func);
            break;
        case amgclCoarseningSmoothedAggrEMin:
            process_sdd<
                Backend,
                amgcl::coarsening::smoothed_aggr_emin<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, solver, direct_solver, func);
            break;
    }
}

//---------------------------------------------------------------------------
template <class Func>
static void process_sdd(
        amgclBackend      backend,
        amgclCoarsening   coarsening,
        amgclRelaxation   relaxation,
        amgclSolver       solver,
        amgclDirectSolver direct_solver,
        const Func &func
        )
{
    switch (backend) {
        case amgclBackendBuiltin:
            process_sdd< amgcl::backend::builtin<double> >(
                    coarsening, relaxation, solver, direct_solver, func);
            break;
        default:
            amgcl::precondition(false, "Unsupported backend");
    }
}

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
struct do_mpi_create {
    amgclHandle          prm;
    MPI_Comm             comm;
    long                 n;
    const long          *ptr;
    const long          *col;
    const double        *val;
    int                  ndv;
    amgclDefVecFunction  dv_func;
    void                *dv_data;

    mutable void *handle;

    do_mpi_create(
            amgclHandle          prm,
            MPI_Comm             comm,
            long                 n,
            const long          *ptr,
            const long          *col,
            const double        *val,
            int                  ndv,
            amgclDefVecFunction  dv_func,
            void                *dv_data
            )
        : prm(prm), comm(comm), n(n), ptr(ptr), col(col), val(val),
          ndv(ndv), dv_func(dv_func), dv_data(dv_data)
    {}

    template <class Solver>
    void process() const {
        using boost::property_tree::ptree;
        using amgcl::detail::empty_ptree;

        ptree *p = static_cast<ptree*>(prm);

        handle = static_cast<void*>(
                new Solver(
                    comm,
                    boost::make_tuple(
                        n,
                        boost::make_iterator_range(ptr, ptr + n + 1),
                        boost::make_iterator_range(col, col + ptr[n]),
                        boost::make_iterator_range(val, val + ptr[n])
                        ),
                    deflation_vectors(ndv, dv_func, dv_data), *p
                    )
                );
    }
};

struct DeflationHandle {
    amgclBackend      backend;
    amgclCoarsening   coarsening;
    amgclRelaxation   relaxation;
    amgclSolver       solver;
    amgclDirectSolver direct_solver;

    void *handle;
    long  n;
};

//---------------------------------------------------------------------------
amgclHandle amgcl_mpi_create(
        amgclBackend         backend,
        amgclCoarsening      coarsening,
        amgclRelaxation      relaxation,
        amgclSolver          solver,
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
    do_mpi_create create(
            params, comm, n, ptr, col, val,
            n_def_vec, def_vec_func, def_vec_data
            );
    process_sdd(backend, coarsening, relaxation, solver, direct_solver, create);

    DeflationHandle *h = new DeflationHandle();

    h->backend       = backend;
    h->coarsening    = coarsening;
    h->relaxation    = relaxation;
    h->solver        = solver;
    h->direct_solver = direct_solver;
    h->handle        = create.handle;
    h->n             = n;

    return static_cast<amgclHandle>(h);
}

struct do_mpi_solve {
    void *handle;
    long  n;

    double const *rhs;
    double       *x;

    do_mpi_solve(void *handle, long n, const double *rhs, double *x)
        : handle(handle), n(n), rhs(rhs), x(x)
    {}

    template <class Solver>
    void process() const {
        Solver *solve = static_cast<Solver*>(handle);

        boost::iterator_range<double const *> rhs_range(rhs, rhs + n);
        boost::iterator_range<double       *> x_range  (x,   x   + n);

        size_t iters;
        double resid;

        boost::tie(iters, resid) = (*solve)(rhs_range, x_range);

        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << resid << std::endl
            << std::endl;
    }
};

//---------------------------------------------------------------------------
void amgcl_mpi_solve(
        amgclHandle   solver,
        double const *rhs,
        double       *x
        )
{
    DeflationHandle *h = static_cast<DeflationHandle*>(solver);

    process_sdd(
            h->backend, h->coarsening, h->relaxation,
            h->solver, h->direct_solver,
            do_mpi_solve(h->handle, h->n, rhs, x)
            );
}

//---------------------------------------------------------------------------
struct do_mpi_destroy {
    void *handle;

    do_mpi_destroy(void* handle) : handle(handle) {}

    template <class Solver>
    void process() const {
        delete static_cast<Solver*>(handle);
    }
};

//---------------------------------------------------------------------------
void amgcl_mpi_destroy(amgclHandle solver) {
    DeflationHandle *h = static_cast<DeflationHandle*>(solver);

    process_sdd(
            h->backend, h->coarsening, h->relaxation,
            h->solver, h->direct_solver,
            do_mpi_destroy(h->handle)
            );

    delete h;
}
