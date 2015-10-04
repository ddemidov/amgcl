#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>

#include <amgcl/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/block_crs.hpp>
#ifdef AMGCL_HAVE_EIGEN
#include <amgcl/backend/eigen.hpp>
#endif
#ifdef AMGCL_HAVE_BLAZE
#include <amgcl/backend/blaze.hpp>
#endif
#ifdef AMGCL_HAVE_VIENNACL
#include <amgcl/backend/viennacl.hpp>
#endif

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

//---------------------------------------------------------------------------
typedef boost::mpl::list<
      amgcl::backend::builtin<double>
    , amgcl::backend::block_crs<double>
#ifdef AMGCL_HAVE_EIGEN
    , amgcl::backend::eigen<double>
#endif
#ifdef AMGCL_HAVE_BLAZE
    , amgcl::backend::blaze<double>
#endif
#ifdef AMGCL_HAVE_VIENNACL
    , amgcl::backend::viennacl< viennacl::compressed_matrix<double> >
    , amgcl::backend::viennacl< viennacl::ell_matrix<double> >
    , amgcl::backend::viennacl< viennacl::hyb_matrix<double> >
#endif
    > backend_list;

//---------------------------------------------------------------------------
template <class Backend, class Matrix>
void test_solver(
        const Matrix &A,
        boost::shared_ptr<typename Backend::vector> const &f,
        boost::shared_ptr<typename Backend::vector>       &x,
        amgcl::runtime::solver::type     solver,
        amgcl::runtime::relaxation::type relaxation,
        amgcl::runtime::coarsening::type coarsening
        )
{
    boost::property_tree::ptree prm;
    prm.put("precond.coarsening.type", coarsening);
    prm.put("precond.relaxation.type", relaxation);
    prm.put("solver.type",             solver);

    amgcl::make_solver<
        amgcl::runtime::amg<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > solve(A, prm);

    std::cout << solve.precond() << std::endl;

    size_t iters;
    double resid;

    amgcl::backend::clear(*x);

    boost::tie(iters, resid) = solve(*f, *x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    BOOST_CHECK_SMALL(resid, 1e-4);
}

//---------------------------------------------------------------------------
template <class Backend, class Matrix>
void test_rap(
        const Matrix &A,
        boost::shared_ptr<typename Backend::vector> const &f,
        boost::shared_ptr<typename Backend::vector>       &x,
        amgcl::runtime::solver::type     solver,
        amgcl::runtime::relaxation::type relaxation
        )
{
    boost::property_tree::ptree prm;
    prm.put("precond.type", relaxation);
    prm.put("solver.type",  solver);

    amgcl::make_solver<
        amgcl::runtime::relaxation::as_preconditioner<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > solve(A, prm);

    std::cout << "Using " << relaxation << " as preconditioner" << std::endl;

    size_t iters;
    double resid;

    amgcl::backend::clear(*x);

    boost::tie(iters, resid) = solve(*f, *x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    BOOST_CHECK_SMALL(resid, 1e-4);
}

//---------------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE_TEMPLATE(test_backends, Backend, backend_list)
{
    amgcl::runtime::coarsening::type coarsening[] = {
        amgcl::runtime::coarsening::ruge_stuben,
        amgcl::runtime::coarsening::aggregation,
        amgcl::runtime::coarsening::smoothed_aggregation,
        amgcl::runtime::coarsening::smoothed_aggr_emin
    };

    amgcl::runtime::relaxation::type relaxation[] = {
        amgcl::runtime::relaxation::gauss_seidel,
        amgcl::runtime::relaxation::multicolor_gauss_seidel,
        amgcl::runtime::relaxation::ilu0,
        amgcl::runtime::relaxation::damped_jacobi,
        amgcl::runtime::relaxation::spai0,
        amgcl::runtime::relaxation::spai1,
        amgcl::runtime::relaxation::chebyshev
    };

    amgcl::runtime::solver::type solver[] = {
        amgcl::runtime::solver::cg,
        amgcl::runtime::solver::bicgstab,
        amgcl::runtime::solver::bicgstabl,
        amgcl::runtime::solver::gmres
    };

    typedef typename Backend::value_type value_type;
    typedef typename Backend::vector     vector;

    std::vector<int>        ptr;
    std::vector<int>        col;
    std::vector<value_type> val;
    std::vector<value_type> rhs;

    size_t n = sample_problem(25, val, col, ptr, rhs);

    typename Backend::params prm;

    boost::shared_ptr<vector> y = Backend::copy_vector(rhs, prm);
    boost::shared_ptr<vector> x = Backend::create_vector(n, prm);

    BOOST_FOREACH(amgcl::runtime::solver::type s, solver) {
        BOOST_FOREACH(amgcl::runtime::relaxation::type r, relaxation) {
            std::cout << Backend::name() << " " << s << " " << r << std::endl;

            try {
                test_rap<Backend>(boost::tie(n, ptr, col, val), y, x, s, r);
            } catch(const std::logic_error&) {}

            BOOST_FOREACH(amgcl::runtime::coarsening::type c, coarsening) {
                std::cout
                    << Backend::name() << " "
                    << s << " " << r << " " << c << std::endl;

                try {
                    test_solver<Backend>(boost::tie(n, ptr, col, val), y, x, s, r, c);
                } catch(const std::logic_error&) {}
            }
        }
    }

}

BOOST_AUTO_TEST_SUITE_END()
