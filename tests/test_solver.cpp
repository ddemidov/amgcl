#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

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

#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggr_emin.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>

#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>

#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/gmres.hpp>

#include "sample_problem.hpp"

//---------------------------------------------------------------------------
typedef boost::mpl::list<
    amgcl::backend::block_crs<float>
    , amgcl::backend::block_crs<double>
#ifdef AMGCL_HAVE_EIGEN
    , amgcl::backend::eigen<float>
    , amgcl::backend::eigen<double>
#endif
#ifdef AMGCL_HAVE_BLAZE
    , amgcl::backend::blaze<float>
    , amgcl::backend::blaze<double>
#endif
#ifdef AMGCL_HAVE_VIENNACL
    , amgcl::backend::viennacl< viennacl::compressed_matrix<float> >
    , amgcl::backend::viennacl< viennacl::compressed_matrix<double> >
    , amgcl::backend::viennacl< viennacl::ell_matrix<float> >
    , amgcl::backend::viennacl< viennacl::ell_matrix<double> >
    , amgcl::backend::viennacl< viennacl::hyb_matrix<float> >
    , amgcl::backend::viennacl< viennacl::hyb_matrix<double> >
#endif
    > cpu_backend_list;

//---------------------------------------------------------------------------
typedef boost::mpl::list<
        amgcl::coarsening::aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::coarsening::smoothed_aggr_emin<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::coarsening::ruge_stuben
    > coarsening_list;

//---------------------------------------------------------------------------
template <
    class                         Backend,
    class                         Coarsening,
    template <class> class        Relax,
    template <class, class> class Solver
    >
void test_solver(const typename Backend::params &prm = typename Backend::params())
{
    typedef typename Backend::value_type value_type;
    typedef typename Backend::vector     vector;

    std::vector<int>        ptr;
    std::vector<int>        col;
    std::vector<value_type> val;
    std::vector<value_type> rhs;

    size_t n = sample_problem(20, val, col, ptr, rhs);

    typedef amgcl::make_solver<Backend, Coarsening, Relax, Solver> AMG;
    typename AMG::AMG_params amg_params;
    amg_params.backend = prm;

    AMG solve(boost::tie(n, ptr, col, val), amg_params);

    std::cout << solve << std::endl;

    boost::shared_ptr<vector> y = Backend::copy_vector(rhs, prm);
    boost::shared_ptr<vector> x = Backend::create_vector(n, prm);

    amgcl::backend::clear(*x);

    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(*y, *x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    if (!(resid < 1e-4)) {
        std::cout << "fuck" << std::endl;
    }

    BOOST_CHECK(resid < 1e-4);
}

//---------------------------------------------------------------------------
template <class Backend, class Coarsening, template <class> class Relax>
void solver_iterator() {
    test_solver<Backend, Coarsening, Relax, amgcl::solver::cg      >();
    test_solver<Backend, Coarsening, Relax, amgcl::solver::bicgstab>();
    test_solver<Backend, Coarsening, Relax, amgcl::solver::gmres   >();
};

//---------------------------------------------------------------------------
template <class Backend, class Enable = void>
struct coarsening_iterator {
    template <class Coarsening>
    void operator()(const Coarsening&) const {
        solver_iterator<Backend, Coarsening, amgcl::relaxation::chebyshev>();
        solver_iterator<Backend, Coarsening, amgcl::relaxation::damped_jacobi>();
        solver_iterator<Backend, Coarsening, amgcl::relaxation::spai0>();
    }
};

//---------------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE( test_solvers )

BOOST_AUTO_TEST_CASE_TEMPLATE(cpu_solvers, Backend, cpu_backend_list)
{
    boost::mpl::for_each<coarsening_list>(
            coarsening_iterator<Backend>()
            );
}

BOOST_AUTO_TEST_CASE(test_serial)
{
    typedef amgcl::backend::builtin<double> Backend;

    test_solver<
        Backend,
        amgcl::coarsening::aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg
        >();

    test_solver<
        Backend,
        amgcl::coarsening::aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::ilu0,
        amgcl::solver::cg
        >();
}

BOOST_AUTO_TEST_SUITE_END()
