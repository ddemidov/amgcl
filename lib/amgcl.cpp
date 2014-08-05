#include <iostream>

#include <boost/type_traits.hpp>

#include <amgcl/amgcl.hpp>

#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/block_crs.hpp>

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

#include "amgcl.h"

#ifdef AMGCL_PROFILING
#include <amgcl/profiler.hpp>
namespace amgcl {
    profiler<> prof;
}
#endif

//---------------------------------------------------------------------------
amgclHandle amgcl_params_create() {
    using boost::property_tree::ptree;
    ptree *p = new ptree();
    return static_cast<amgclHandle>(p);
}

//---------------------------------------------------------------------------
void amgcl_params_seti(amgclHandle prm, const char *name, int value) {
    using boost::property_tree::ptree;
    ptree *p = static_cast<ptree*>(prm);
    p->put(name, value);
}

//---------------------------------------------------------------------------
void amgcl_params_setf(amgclHandle prm, const char *name, float value) {
    using boost::property_tree::ptree;
    ptree *p = static_cast<ptree*>(prm);
    p->put(name, value);
}

//---------------------------------------------------------------------------
void amgcl_params_destroy(amgclHandle prm) {
    using boost::property_tree::ptree;
    delete static_cast<ptree*>(prm);
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    template <class> class Relaxation,
    class Func
    >
static void process_precond(const Func &func)
{
    typedef amgcl::amg<Backend, Coarsening, Relaxation> AMG;
    func.template process<AMG>();
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
process_precond(amgclRelaxation relaxation, const Func &func)
{
    switch (relaxation) {
        case amgclRelaxationDampedJacobi:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::damped_jacobi
                >(func);
            break;
        case amgclRelaxationGaussSeidel:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::gauss_seidel
                >(func);
            break;
        case amgclRelaxationChebyshev:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::chebyshev
                >(func);
            break;
        case amgclRelaxationSPAI0:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::spai0
                >(func);
            break;
        case amgclRelaxationILU0:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::ilu0
                >(func);
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
process_precond(amgclRelaxation relaxation, const Func &func)
{
    switch (relaxation) {
        case amgclRelaxationDampedJacobi:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::damped_jacobi
                >(func);
            break;
        case amgclRelaxationChebyshev:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::chebyshev
                >(func);
            break;
        case amgclRelaxationSPAI0:
            process_precond<
                Backend,
                Coarsening,
                amgcl::relaxation::spai0
                >(func);
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
static void process_precond(
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        const Func &func
        )
{
    switch (coarsening) {
        case amgclCoarseningRugeStuben:
            process_precond<
                Backend,
                amgcl::coarsening::ruge_stuben
                >(relaxation, func);
            break;
        case amgclCoarseningAggregation:
            process_precond<
                Backend,
                amgcl::coarsening::aggregation<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, func);
            break;
        case amgclCoarseningSmoothedAggregation:
            process_precond<
                Backend,
                amgcl::coarsening::smoothed_aggregation<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, func);
            break;
        case amgclCoarseningSmoothedAggrEMin:
            process_precond<
                Backend,
                amgcl::coarsening::smoothed_aggr_emin<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, func);
            break;
    }
}

//---------------------------------------------------------------------------
template <class Func>
static void process_precond(
        amgclBackend    backend,
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        const Func &func
        )
{
    switch (backend) {
        case amgclBackendBuiltin:
            process_precond< amgcl::backend::builtin<double> >(
                    coarsening, relaxation, func);
            break;
        case amgclBackendBlockCRS:
            process_precond< amgcl::backend::block_crs<double> >(
                    coarsening, relaxation, func);
            break;
    }
}

//---------------------------------------------------------------------------
struct AMGHandle {
    amgclBackend    backend;
    amgclCoarsening coarsening;
    amgclRelaxation relaxation;

    void *handle;
};

//---------------------------------------------------------------------------
struct do_create_precond {
    amgclHandle prm;

    int  n;

    const int    *ptr;
    const int    *col;
    const double *val;

    mutable void *handle;

    do_create_precond(
            amgclHandle prm,
            int  n,
            const int    *ptr,
            const int    *col,
            const double *val
          )
        : prm(prm), n(n), ptr(ptr), col(col), val(val), handle(0)
    {}

    template <class AMG>
    void process() const {
        using boost::property_tree::ptree;
        const ptree *p = static_cast<const ptree*>(prm);

        AMG *amg = new AMG(
                boost::make_tuple(
                    n,
                    boost::make_iterator_range(ptr, ptr + n + 1),
                    boost::make_iterator_range(col, col + ptr[n]),
                    boost::make_iterator_range(val, val + ptr[n])
                    ),
                *p
                );

        std::cout << *amg << std::endl;

        handle = static_cast<void*>(amg);
    }
};

//---------------------------------------------------------------------------
amgclHandle amgcl_precond_create(
        amgclBackend    backend,
        amgclCoarsening coarsening,
        amgclRelaxation relaxation,
        amgclHandle     prm,
        int n,
        const int    *ptr,
        const int    *col,
        const double *val
        )
{
    do_create_precond create(prm, n, ptr, col, val);
    process_precond(backend, coarsening, relaxation, create);

    AMGHandle *h = new AMGHandle();

    h->backend    = backend;
    h->coarsening = coarsening;
    h->relaxation = relaxation;
    h->handle     = create.handle;

    return static_cast<amgclHandle>(h);
}

//---------------------------------------------------------------------------
struct do_apply_precond {
    void *handle;
    double const *rhs;
    double       *x;

    do_apply_precond(void *handle, const double *rhs, double *x)
        : handle(handle), rhs(rhs), x(x) {}

    template <class AMG>
    void process() const {
        AMG *amg = static_cast<AMG*>(handle);
        const size_t n = amgcl::backend::rows( amg->top_matrix() );

        boost::iterator_range<const double*> rhs_range(rhs, rhs + n);
        boost::iterator_range<double*> x_range(x, x + n);

        amg->apply(rhs_range, x_range);
    }
};

//---------------------------------------------------------------------------
void amgcl_precond_apply(amgclHandle handle, const double *rhs, double *x) {
    AMGHandle *h = static_cast<AMGHandle*>(handle);

    process_precond(
            h->backend, h->coarsening, h->relaxation,
            do_apply_precond(h->handle, rhs, x)
            );
}

//---------------------------------------------------------------------------
struct do_destroy_precond {
    void *handle;

    do_destroy_precond(void *handle) : handle(handle) {}

    template <class AMG>
    void process() const {
        delete static_cast<AMG*>(handle);
    }
};

//---------------------------------------------------------------------------
void amgcl_precond_destroy(amgclHandle handle) {
    AMGHandle *h = static_cast<AMGHandle*>(handle);

    process_precond(
            h->backend, h->coarsening, h->relaxation,
            do_destroy_precond(h->handle)
            );

    delete h;
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Func
    >
static void process_solver(amgclSolver solver, const Func &func)
{
    switch (solver) {
        case amgclSolverCG:
            func.template process< amgcl::solver::cg<Backend> >();
            break;
        case amgclSolverBiCGStab:
            func.template process< amgcl::solver::bicgstab<Backend> >();
            break;
        case amgclSolverBiCGStabL:
            func.template process< amgcl::solver::bicgstabl<Backend> >();
            break;
        case amgclSolverGMRES:
            func.template process< amgcl::solver::gmres<Backend> >();
            break;
    }
}

//---------------------------------------------------------------------------
template <class Func>
static void process_solver(
        amgclBackend backend,
        amgclSolver  solver,
        const Func &func
        )
{
    switch (backend) {
        case amgclBackendBuiltin:
            process_solver< amgcl::backend::builtin<double> >(solver, func);
            break;
        case amgclBackendBlockCRS:
            process_solver< amgcl::backend::block_crs<double> >(solver, func);
            break;
    }
}

//---------------------------------------------------------------------------
struct do_create_solver {
    amgclHandle prm;
    int n;

    mutable void *handle;

    do_create_solver(amgclHandle prm, int n) : prm(prm), n(n), handle(0) {}

    template <class Solver>
    void process() const {
        using boost::property_tree::ptree;
        ptree *p = static_cast<ptree*>(prm);

        handle = static_cast<void*>(new Solver(n, *p));
    }
};

//---------------------------------------------------------------------------
struct SolverHandle {
    amgclBackend backend;
    amgclSolver  solver;

    void *handle;
};

//---------------------------------------------------------------------------
amgclHandle amgcl_solver_create(
        amgclBackend backend,
        amgclSolver  solver,
        amgclHandle  prm,
        int n
        )
{
    do_create_solver create(prm, n);
    process_solver(backend, solver, create);

    SolverHandle *h = new SolverHandle();
    h->backend = backend;
    h->solver  = solver;
    h->handle  = create.handle;

    return static_cast<amgclHandle>(h);
}

//---------------------------------------------------------------------------
struct do_destroy_solver {
    void *handle;

    do_destroy_solver(void* handle) : handle(handle) {}

    template <class Solver>
    void process() const {
        delete static_cast<Solver*>(handle);
    }
};

//---------------------------------------------------------------------------
void amgcl_solver_destroy(amgclHandle handle) {
    SolverHandle *h = static_cast<SolverHandle*>(handle);

    process_solver(h->backend, h->solver, do_destroy_solver(h->handle));

    delete h;
}

//---------------------------------------------------------------------------
struct do_solve {
    void *slv_handle;
    void *amg_handle;

    const double *rhs;
    double *x;

    do_solve(void *slv_handle, void *amg_handle, const double *rhs, double *x)
        : slv_handle(slv_handle), amg_handle(amg_handle), rhs(rhs), x(x)
    {}

    template <class Solver>
    struct call_solver {
        Solver *solve;
        void *amg_handle;

        const double *rhs;
        double *x;

        call_solver(Solver *solve, void *amg_handle, const double *rhs, double *x)
            : solve(solve), amg_handle(amg_handle), rhs(rhs), x(x)
        {}

        template <class AMG>
        void process() const {
            AMG *amg = static_cast<AMG*>(amg_handle);
            const size_t n = amgcl::backend::rows( amg->top_matrix() );

            boost::iterator_range<const double*> rhs_range(rhs, rhs + n);
            boost::iterator_range<double*> x_range(x, x + n);

            size_t iters;
            double resid;

            boost::tie(iters, resid) = (*solve)(*amg, rhs_range, x_range);

            std::cout
                << "Iterations: " << iters << std::endl
                << "Error:      " << resid << std::endl
                << std::endl;
        }
    };

    template <class Solver>
    void process() const {
        AMGHandle *amg    = static_cast<AMGHandle*>(amg_handle);
        Solver    *solver = static_cast<Solver*   >(slv_handle);

        process_precond(amg->backend, amg->coarsening, amg->relaxation,
                call_solver<Solver>(solver, amg->handle, rhs, x));
    }
};

//---------------------------------------------------------------------------
void amgcl_solver_solve(
        amgclHandle solver,
        amgclHandle amg,
        const double *rhs,
        double *x
        )
{
    SolverHandle *h = static_cast<SolverHandle*>(solver);

    process_solver(h->backend, h->solver,
            do_solve(h->handle, amg, rhs, x)
            );
}

//---------------------------------------------------------------------------
struct do_solve_mtx {
    void *slv_handle;
    void *amg_handle;

    const int    * A_ptr;
    const int    * A_col;
    const double * A_val;

    const double * rhs;
    double * x;

    do_solve_mtx(
            void *slv_handle, void *amg_handle,
            const int * A_ptr, const int * A_col, const double * A_val,
            const double *rhs, double *x
            )
        : slv_handle(slv_handle), amg_handle(amg_handle),
          A_ptr(A_ptr), A_col(A_col), A_val(A_val),
          rhs(rhs), x(x)
    {}

    template <class Solver>
    struct call_solver {
        Solver *solve;
        void *amg_handle;

        const int    * A_ptr;
        const int    * A_col;
        const double * A_val;

        const double *rhs;
        double *x;

        call_solver(
                Solver *solve, void *amg_handle,
                const int * A_ptr, const int * A_col, const double * A_val,
                const double *rhs, double *x
                )
            : solve(solve), amg_handle(amg_handle),
              A_ptr(A_ptr), A_col(A_col), A_val(A_val),
              rhs(rhs), x(x)
        {}

        template <class AMG>
        void process() const {
            AMG *amg = static_cast<AMG*>(amg_handle);
            const size_t n = amgcl::backend::rows( amg->top_matrix() );

            boost::iterator_range<const double*> rhs_range(rhs, rhs + n);
            boost::iterator_range<double*> x_range(x, x + n);

            size_t iters;
            double resid;

            boost::tie(iters, resid) = (*solve)(
                    boost::make_tuple(
                        n,
                        boost::make_iterator_range(A_ptr, A_ptr + n + 1),
                        boost::make_iterator_range(A_col, A_col + A_ptr[n]),
                        boost::make_iterator_range(A_val, A_val + A_ptr[n])
                        ),
                    *amg, rhs_range, x_range
                    );

            std::cout
                << "Iterations: " << iters << std::endl
                << "Error:      " << resid << std::endl
                << std::endl;
        }
    };

    template <class Solver>
    void process() const {
        AMGHandle *amg    = static_cast<AMGHandle*>(amg_handle);
        Solver    *solver = static_cast<Solver*   >(slv_handle);

        process_precond(amg->backend, amg->coarsening, amg->relaxation,
                call_solver<Solver>(
                    solver, amg->handle, A_ptr, A_col, A_val, rhs, x
                    )
                );
    }
};

//---------------------------------------------------------------------------
void amgcl_solver_solve_mtx(
        amgclHandle solver,
        int    const * A_ptr,
        int    const * A_col,
        double const * A_val,
        amgclHandle amg,
        const double *rhs,
        double *x
        )
{
    SolverHandle *h = static_cast<SolverHandle*>(solver);

    process_solver(h->backend, h->solver,
            do_solve_mtx(h->handle, amg, A_ptr, A_col, A_val, rhs, x)
            );
}
