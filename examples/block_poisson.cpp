#include <iostream>
#include <string>
#include <random>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>
#include <boost/range/iterator_range.hpp>

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>

#include <amgcl/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/profiler.hpp>

template <int N, int M>
using block = amgcl::static_matrix<double, N, M>;

//---------------------------------------------------------------------------
template <int N>
block<N,N> make_block(bool diagonal = false) {
    block<N,N> b = amgcl::math::zero<block<N,N>>();

    for(int i = 0; i < N; ++i)
        b(i,i) = diagonal ? 6 : -1;

    return b;
}

template <int B>
int sample_problem(
        int n,
        std::vector<int>        &ptr,
        std::vector<int>        &col,
        std::vector<block<B,B>> &val
        )
{
    int n3  = n * n * n;

    ptr.clear();
    col.clear();
    val.clear();

    ptr.reserve(n3 + 1);
    col.reserve(n3 * 7);
    val.reserve(n3 * 7);

    ptr.push_back(0);
    for(int k = 0, idx = 0; k < n; ++k) {
        for(int j = 0; j < n; ++j) {
            for(int i = 0; i < n; ++i, ++idx) {
                if (k > 0) {
                    col.push_back(idx - n * n);
                    val.push_back(make_block<B>());
                }

                if (j > 0) {
                    col.push_back(idx - n);
                    val.push_back(make_block<B>());
                }

                if (i > 0) {
                    col.push_back(idx - 1);
                    val.push_back(make_block<B>());
                }

                col.push_back(idx);
                val.push_back(make_block<B>(true));

                if (i + 1 < n) {
                    col.push_back(idx + 1);
                    val.push_back(make_block<B>());
                }

                if (j + 1 < n) {
                    col.push_back(idx + n);
                    val.push_back(make_block<B>());
                }

                if (k + 1 < n) {
                    col.push_back(idx + n * n);
                    val.push_back(make_block<B>());
                }

                ptr.push_back(col.size());
            }
        }
    }

    return n3;
}

//---------------------------------------------------------------------------
template <int B>
void solve(int m, const boost::property_tree::ptree &prm) {
    namespace math = amgcl::math;

    amgcl::profiler<> prof;

    std::vector<int> ptr, col;
    std::vector<block<B,B>> val;

    int n = sample_problem<B>(m, ptr, col, val);

    typedef amgcl::backend::vexcl<block<B,B>> Backend;

    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

#if defined(VEXCL_BACKEND_CUDA)
    vex::push_compile_options(ctx, "-Xcompiler -std=c++03");
#endif

    amgcl::backend::enable_static_matrix_for_vexcl(ctx);

    typename Backend::params bprm;
    bprm.q = ctx;

    typedef amgcl::make_solver<
        amgcl::runtime::amg<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > Solver;

    prof.tic("setup");
    Solver solve(boost::tie(n, ptr, col, val), prm, bprm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    vex::vector< block<B,1> > f(ctx, n);
    vex::vector< block<B,1> > x(ctx, n);

    f = math::constant< block<B,1> >(1.0);
    x = math::constant< block<B,1> >(0.0);

    prof.tic("solve");
    int    iters;
    double error;
    boost::tie(iters, error) = solve(f, x);
    prof.toc("solve");

    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;

    using std::vector;
    using std::string;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        ("prm-file,P",
         po::value<string>(),
         "Parameter file in json format. "
        )
        (
         "prm,p",
         po::value< vector<string> >(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        (
         "block-size,b",
         po::value<int>()->default_value(4),
         "The block size of the system matrix. "
         "When specified, the system matrix is assumed to have block-wise structure. "
         "This usually is the case for problems in elasticity, structural mechanics, "
         "for coupled systems of PDE (such as Navier-Stokes equations), etc. "
         "Valid choices are 2, 3, 4, and 6."
        )
        (
         "size,n",
         po::value<int>()->default_value(50),
         "The size of the Poisson problem to solve when no system matrix is given. "
         "Specified as number of grid nodes along each dimension of a unit cube. "
         "The resulting system will have n*n*n unknowns. "
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    boost::property_tree::ptree prm;
    if (vm.count("prm-file")) {
        read_json(vm["prm-file"].as<string>(), prm);
    }

    if (vm.count("prm")) {
        BOOST_FOREACH(string pair, vm["prm"].as<vector<string> >())
        {
            using namespace boost::algorithm;
            vector<string> key_val;
            split(key_val, pair, is_any_of("="));
            if (key_val.size() != 2) throw po::invalid_option_value(
                    "Parameters specified with -p option "
                    "should have name=value format");

            prm.put(key_val[0], key_val[1]);
        }
    }

    switch(vm["block-size"].as<int>()) {

        case 2:
            solve<2>(vm["size"].as<int>(), prm);
            break;
        case 3:
            solve<3>(vm["size"].as<int>(), prm);
            break;
        case 4:
            solve<4>(vm["size"].as<int>(), prm);
            break;
        case 8:
            solve<8>(vm["size"].as<int>(), prm);
            break;
        default:
            throw std::logic_error("Unsupported block size");
    }
}
