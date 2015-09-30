#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>

#include <amgcl/runtime.hpp>
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
      amgcl::backend::block_crs<double>
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
template <class Backend>
void test_solver(
        amgcl::runtime::coarsening::type coarsening,
        amgcl::runtime::relaxation::type relaxation,
        amgcl::runtime::solver::type     solver
        )
{
    typedef typename Backend::value_type value_type;
    typedef typename Backend::vector     vector;


    std::vector<int>        ptr;
    std::vector<int>        col;
    std::vector<value_type> val;
    std::vector<value_type> rhs;

    size_t n = sample_problem(25, val, col, ptr, rhs);

    typedef amgcl::runtime::make_solver<Backend> Solver;

    boost::property_tree::ptree prm;
    prm.put("amg.coarsening.type", coarsening);
    prm.put("amg.relaxation.type", relaxation);
    prm.put("solver.type",         solver);

    Solver solve(boost::tie(n, ptr, col, val), prm);

    std::cout << solve.amg() << std::endl;

    boost::shared_ptr<vector> y = Backend::copy_vector(rhs, boost::property_tree::ptree());
    boost::shared_ptr<vector> x = Backend::create_vector(n, boost::property_tree::ptree());

    amgcl::backend::clear(*x);

    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(*y, *x);

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

    BOOST_FOREACH(amgcl::runtime::coarsening::type c, coarsening) {
        BOOST_FOREACH(amgcl::runtime::relaxation::type r, relaxation) {
            BOOST_FOREACH(amgcl::runtime::solver::type s, solver) {
                std::cout
                    << Backend::name() << " "
                    << c << " " << r << " " << s << std::endl;

                try {
                    test_solver<Backend>(c, r, s);
                } catch(const std::logic_error&) {}
            }
        }
    }

}

BOOST_AUTO_TEST_SUITE_END()
