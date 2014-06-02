#define BOOST_TEST_MODULE MatrixCRS
#include <set>

#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/range.hpp>
#include <boost/range/combine.hpp>
#include <boost/foreach.hpp>
#include <boost/mpl/list.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/block_crs.hpp>
#include <amgcl/backend/ccrs.hpp>
#ifdef AMGCL_HAVE_EIGEN
#include <amgcl/backend/eigen.hpp>
#endif
#ifdef AMGCL_HAVE_VEXCL
#include <amgcl/backend/vexcl.hpp>
#endif

template <typename P, typename C, typename V>
void random_problem(size_t n, size_t m, size_t nnz_per_row,
        std::vector<P> &ptr,
        std::vector<C> &col,
        std::vector<V> &val,
        std::vector<V> &vec
        )
{
    ptr.reserve(n + 1);
    col.reserve(nnz_per_row * n);

    boost::random::mt19937 rng;
    boost::random::uniform_int_distribution<C> random_column(0, m - 1);

    ptr.push_back(0);
    for(size_t i = 0; i < n; i++) {
        std::set<C> cs;
        for(size_t j = 0; j < nnz_per_row; ++j)
            cs.insert(random_column(rng));

        BOOST_FOREACH(C c, cs) col.push_back(c);

        ptr.push_back(static_cast<P>( col.size() ));
    }

    boost::random::uniform_real_distribution<V> random_value(0, 1);

    val.resize( col.size() );
    BOOST_FOREACH(V &v, val) v = random_value(rng);

    vec.resize(n);
    BOOST_FOREACH(V &v, vec) v = random_value(rng);
}

template <class Backend>
void test_backend(typename Backend::params const prm = typename Backend::params())
{
    typedef typename Backend::value_type V;
    typedef typename Backend::index_type I;

    typedef typename Backend::matrix  matrix;
    typedef typename Backend::vector  vector;

    typedef amgcl::backend::crs<V, I> ref_matrix;

    const size_t n = 256;

    std::vector<I> ptr;
    std::vector<I> col;
    std::vector<V> val;
    std::vector<V> vec;

    random_problem(n, n, 16, ptr, col, val, vec);

    boost::shared_ptr<ref_matrix> Aref = boost::make_shared<ref_matrix>(n, n, ptr, col, val);
    boost::shared_ptr<matrix>     Atst = Backend::copy_matrix(Aref, prm);

    boost::shared_ptr<vector> x = Backend::copy_vector(vec, prm);
    boost::shared_ptr<vector> y = Backend::create_vector(n, prm);

    amgcl::backend::clear(*y);

    std::vector<V> y_ref(n, 0);

    amgcl::backend::spmv(2, *Aref, vec, 1, y_ref);
    amgcl::backend::spmv(2, *Atst, *x,  1, *y);

    for(size_t i = 0; i < n; ++i)
        BOOST_CHECK_CLOSE(static_cast<V>((*y)[i]), y_ref[i], 1e-4);
}

BOOST_AUTO_TEST_SUITE( backend_crs )

typedef boost::mpl::list<
    amgcl::backend::block_crs<float>
    , amgcl::backend::block_crs<double>
    , amgcl::backend::compressed_crs<float>
    , amgcl::backend::compressed_crs<double>
#ifdef AMGCL_HAVE_EIGEN
    , amgcl::backend::eigen<float>
    , amgcl::backend::eigen<double>
#endif
    > backends;

BOOST_AUTO_TEST_CASE_TEMPLATE(construct, Backend, backends)
{
    test_backend<Backend>();
}

#ifdef AMGCL_HAVE_VEXCL
typedef boost::mpl::list<
    amgcl::backend::vexcl<float>,
    amgcl::backend::vexcl<double>
    > vexcl_backends;

BOOST_AUTO_TEST_CASE_TEMPLATE(construct_vexcl, Backend, vexcl_backends)
{
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    std::cout << ctx << std::endl;

    typename Backend::params prm;
    prm.q = ctx;

    test_backend<Backend>(prm);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
