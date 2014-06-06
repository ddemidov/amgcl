#define BOOST_TEST_MODULE TestSolvers
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>
#include <boost/mpl/for_each.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/block_crs.hpp>

#ifdef AMGCL_HAVE_EIGEN
#include <amgcl/backend/eigen.hpp>
#endif

#ifdef AMGCL_HAVE_VIENNACL
#include <amgcl/backend/viennacl.hpp>
#endif

#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>

#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai.hpp>
#include <amgcl/relaxation/chebyshev.hpp>

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
            >
    > coarsening_list;

//---------------------------------------------------------------------------
typedef boost::mpl::list<
    boost::mpl::integral_c<amgcl::relaxation::scheme, amgcl::relaxation::damped_jacobi>,
    boost::mpl::integral_c<amgcl::relaxation::scheme, amgcl::relaxation::spai0>,
    boost::mpl::integral_c<amgcl::relaxation::scheme, amgcl::relaxation::chebyshev>
    > relax_list;

//---------------------------------------------------------------------------
template <
    class                     Backend,
    class                     Coarsening,
    amgcl::relaxation::scheme Relax,
    class                     Solver
    >
void test_solver(const typename Backend::params &prm = typename Backend::params())
{
    typedef typename Backend::value_type value_type;
    typedef typename Backend::vector     vector;

    amgcl::backend::crs<value_type, int> A;
    std::vector<value_type> rhs;

    size_t n = A.nrows = A.ncols = sample_problem(20, A.val, A.col, A.ptr, rhs);

    amgcl::amg<Backend, Coarsening, Relax> amg(A);
    std::cout << amg << std::endl;

    boost::shared_ptr<vector> y = Backend::copy_vector(rhs, prm);
    boost::shared_ptr<vector> x = Backend::create_vector(n, prm);

    amgcl::backend::clear(*x);

    Solver solve(n, prm);

    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(amg.top_matrix(), *y, amg, *x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    BOOST_CHECK(resid < 1e-4);
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    amgcl::relaxation::scheme Relax,
    template <class> class Solver
    >
void run_test() {
    test_solver<
        Backend,
        Coarsening,
        Relax,
        Solver<Backend>
        >();
}

//---------------------------------------------------------------------------
template <class Backend, class Coarsening>
struct relaxation_iterator {
    template <class Relax>
    void operator()(const Relax&) const {
        run_test<Backend, Coarsening, Relax::value, amgcl::solver::cg      >();
        run_test<Backend, Coarsening, Relax::value, amgcl::solver::bicgstab>();
        run_test<Backend, Coarsening, Relax::value, amgcl::solver::gmres   >();
    }
};

//---------------------------------------------------------------------------
template <class Backend, class Enable = void>
struct coarsening_iterator {
    template <class Coarsening>
    void operator()(const Coarsening&) const {
        boost::mpl::for_each<relax_list>(
                relaxation_iterator<Backend, Coarsening>()
                );
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

BOOST_AUTO_TEST_CASE(test_gauss_seidel)
{
    typedef amgcl::backend::builtin<double> Backend;

    test_solver<
        Backend,
        amgcl::coarsening::aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::gauss_seidel,
        amgcl::solver::cg<Backend>
        >();
}

BOOST_AUTO_TEST_SUITE_END()
