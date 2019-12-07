#ifndef TESTS_TEST_SOLVER_HPP
#define TESTS_TEST_SOLVER_HPP

#include <amgcl/amg.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/zero_copy.hpp>
#include <amgcl/profiler.hpp>

#include <boost/assign/std/vector.hpp>
using namespace boost::assign;

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

//---------------------------------------------------------------------------
template <class Backend, class Matrix>
void test_solver(
        const Matrix &A,
        std::shared_ptr<typename Backend::vector> const &f,
        std::shared_ptr<typename Backend::vector>       &x,
        amgcl::runtime::solver::type     solver,
        amgcl::runtime::relaxation::type relaxation,
        amgcl::runtime::coarsening::type coarsening,
        bool test_null_space = false
        )
{
    boost::property_tree::ptree prm;
    prm.put("precond.coarse_enough",   500);
    prm.put("precond.coarsening.type", coarsening);
    prm.put("precond.relax.type",      relaxation);
    prm.put("solver.type",             solver);

    typedef typename Backend::value_type value_type;
    std::vector<value_type> null;

    if (test_null_space) {
        size_t n = amgcl::backend::rows(*A);
        null.resize(n, amgcl::math::constant<value_type>(1));

        prm.put("precond.coarsening.nullspace.cols", 1);
        prm.put("precond.coarsening.nullspace.rows", n);
        prm.put("precond.coarsening.nullspace.B",    &null[0]);
    }

    amgcl::make_solver<
        amgcl::amg<Backend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper>,
        amgcl::runtime::solver::wrapper<Backend>
        > solve(A, prm);

    std::cout << solve.precond() << std::endl;

    size_t iters;
    double resid;

    amgcl::backend::clear(*x);

    std::tie(iters, resid) = solve(*f, *x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    BOOST_REQUIRE_SMALL(resid, 1e-4);
}

//---------------------------------------------------------------------------
template <class Backend, class Matrix>
void test_rap(
        const Matrix &A,
        std::shared_ptr<typename Backend::vector> const &f,
        std::shared_ptr<typename Backend::vector>       &x,
        amgcl::runtime::solver::type     solver,
        amgcl::runtime::relaxation::type relaxation
        )
{
    boost::property_tree::ptree prm;
    prm.put("precond.type", relaxation);
    prm.put("solver.type",  solver);

    amgcl::make_solver<
        amgcl::relaxation::as_preconditioner<Backend, amgcl::runtime::relaxation::wrapper>,
        amgcl::runtime::solver::wrapper<Backend>
        > solve(A, prm);

    std::cout << "Using " << relaxation << " as preconditioner" << std::endl;

    size_t iters;
    double resid;

    amgcl::backend::clear(*x);

    std::tie(iters, resid) = solve(*f, *x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    BOOST_CHECK_SMALL(resid, 1e-4);
}

template <class Backend, class value_type, class rhs_type>
void test_problem(
        size_t n,
        std::vector<ptrdiff_t>  ptr,
        std::vector<ptrdiff_t>  col,
        std::vector<value_type> val,
        std::vector<rhs_type>   rhs
        )
{
    amgcl::runtime::coarsening::type coarsening[] = {
        amgcl::runtime::coarsening::aggregation,
        amgcl::runtime::coarsening::smoothed_aggregation,
        amgcl::runtime::coarsening::smoothed_aggr_emin,
        amgcl::runtime::coarsening::ruge_stuben
    };

    amgcl::runtime::relaxation::type relaxation[] = {
        amgcl::runtime::relaxation::spai0
#ifndef AMGCL_RUNTIME_DISABLE_SPAI1
      , amgcl::runtime::relaxation::spai1
#endif
      , amgcl::runtime::relaxation::damped_jacobi
      , amgcl::runtime::relaxation::gauss_seidel
      , amgcl::runtime::relaxation::ilu0
      , amgcl::runtime::relaxation::iluk
      , amgcl::runtime::relaxation::ilut
#ifndef AMGCL_RUNTIME_DISABLE_CHEBYSHEV
      , amgcl::runtime::relaxation::chebyshev
#endif
    };

    amgcl::runtime::solver::type solver[] = {
        amgcl::runtime::solver::cg,
        amgcl::runtime::solver::bicgstab,
        amgcl::runtime::solver::bicgstabl,
        amgcl::runtime::solver::gmres,
        amgcl::runtime::solver::lgmres,
        amgcl::runtime::solver::fgmres,
        amgcl::runtime::solver::idrs
    };

    typename Backend::params prm;

    auto y = Backend::copy_vector(rhs, prm);
    auto x = Backend::create_vector(n, prm);

    // Test solvers
    for(amgcl::runtime::solver::type s : solver) {
        std::cout << "Solver: " << s << std::endl;
        try {
            test_solver<Backend>(
                    amgcl::adapter::zero_copy(n, ptr.data(), col.data(), val.data()),
                    y, x, s, relaxation[0], coarsening[0]
                    );
        } catch(const std::logic_error&) {}
    }

    // Test smoothers
    for(amgcl::runtime::relaxation::type r : relaxation) {
        std::cout << "Relaxation: " << r << std::endl;
        try {
            test_solver<Backend>(
                    amgcl::adapter::zero_copy(n, ptr.data(), col.data(), val.data()),
                    y, x, solver[0], r, coarsening[0]);
        } catch(const std::logic_error&) {}

        try {
            std::cout << "Relaxation as preconditioner: " << r << std::endl;

            test_rap<Backend>(
                    amgcl::adapter::zero_copy(n, ptr.data(), col.data(), val.data()),
                    y, x, solver[0], r);
        } catch(const std::logic_error&) {}
    }

    // Test coarsening
    for(amgcl::runtime::coarsening::type c : coarsening) {
        std::cout << "Coarsening: " << c << std::endl;

        try {
            test_solver<Backend>(
                    amgcl::adapter::zero_copy(n, ptr.data(), col.data(), val.data()),
                    y, x, solver[0], relaxation[0], c);
        } catch(const std::logic_error&) {}

        switch (c) {
            case amgcl::runtime::coarsening::aggregation:
            case amgcl::runtime::coarsening::smoothed_aggregation:
            case amgcl::runtime::coarsening::smoothed_aggr_emin:
                test_solver<Backend>(
                        amgcl::adapter::zero_copy(n, ptr.data(), col.data(), val.data()),
                        y, x, solver[0], relaxation[0], c, /*test_null_space*/true);
                break;
            default:
                break;
        }
    }
}

template <class Backend>
void test_backend() {
    typedef typename Backend::value_type value_type;
    typedef typename amgcl::math::rhs_of<value_type>::type rhs_type;

    // Poisson 3D
    {
        std::vector<ptrdiff_t>  ptr;
        std::vector<ptrdiff_t>  col;
        std::vector<value_type> val;
        std::vector<rhs_type>   rhs;

        size_t n = sample_problem(32, val, col, ptr, rhs);

        test_problem<Backend>(n, ptr, col, val, rhs);
    }

    // Trivial problem
#if !defined(SOLVER_BACKEND_VIENNACL)
    {
        std::vector<ptrdiff_t>  ptr;
        std::vector<ptrdiff_t>  col;
        std::vector<value_type> val;
        std::vector<rhs_type>   rhs;

	val += amgcl::math::identity<value_type>(), amgcl::math::identity<value_type>();
	col += 0, 1;
	ptr += 0, 1, 2;
	rhs += amgcl::math::constant<rhs_type>(1.0), amgcl::math::zero<rhs_type>();

	size_t n = rhs.size();

        test_problem<Backend>(n, ptr, col, val, rhs);
    }
#endif
}

#endif
