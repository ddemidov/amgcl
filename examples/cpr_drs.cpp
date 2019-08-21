#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#if defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/backend/vexcl.hpp>
#  include <amgcl/backend/vexcl_static_matrix.hpp>
   typedef amgcl::backend::vexcl<double> Backend;
#elif defined(SOLVER_BACKEND_VIENNACL)
#  include <amgcl/backend/viennacl.hpp>
   typedef amgcl::backend::viennacl< viennacl::compressed_matrix<double> > Backend;
#elif defined(SOLVER_BACKEND_CUDA)
#  include <amgcl/backend/cuda.hpp>
#  include <amgcl/relaxation/cusparse_ilu0.hpp>
   typedef amgcl::backend::cuda<double> Backend;
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#  include <amgcl/backend/builtin.hpp>
   typedef amgcl::backend::builtin<double> Backend;
#endif

#if defined(SOLVER_BACKEND_BUILTIN) || defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/value_type/static_matrix.hpp>
#  include <amgcl/adapter/block_matrix.hpp>
#endif

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/preconditioner/cpr_drs.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
template <class Matrix>
void solve_cpr(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    using amgcl::prof;
    Backend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;
    bprm.q = ctx;
#elif defined(SOLVER_BACKEND_VIENNACL)
    std::cout
        << viennacl::ocl::current_device().name()
        << " (" << viennacl::ocl::current_device().vendor() << ")\n\n";
#elif defined(SOLVER_BACKEND_CUDA)
    cusparseCreate(&bprm.cusparse_handle);
    {
        int dev;
        cudaGetDevice(&dev);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << prop.name << std::endl << std::endl;
    }
#endif

    auto t1 = prof.scoped_tic("CPR");

    typedef
        amgcl::amg<Backend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper>
        PPrecond;

    typedef
        amgcl::relaxation::as_preconditioner<Backend, amgcl::runtime::relaxation::wrapper>
        SPrecond;

    prof.tic("setup");
    amgcl::make_solver<
        amgcl::preconditioner::cpr_drs<PPrecond, SPrecond>,
        amgcl::runtime::solver::wrapper<Backend>
        > solve(K, prm, bprm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    auto f = Backend::copy_vector(rhs, bprm);
    auto x = Backend::create_vector(rhs.size(), bprm);
    amgcl::backend::clear(*x);

    size_t iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = solve(*f, *x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
}

#if defined(SOLVER_BACKEND_BUILTIN) || defined(SOLVER_BACKEND_VEXCL)
//---------------------------------------------------------------------------
template <int B, class Matrix>
void solve_block_cpr(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    using amgcl::prof;

    auto t1 = prof.scoped_tic("CPR");

    typedef amgcl::static_matrix<double, B, B> val_type;
    typedef amgcl::static_matrix<double, B, 1> rhs_type;

#if defined(SOLVER_BACKEND_BUILTIN)
    typedef amgcl::backend::builtin<val_type>  SBackend;
    typedef amgcl::backend::builtin<double>    PBackend;
#elif defined(SOLVER_BACKEND_VEXCL)
    typedef amgcl::backend::vexcl<val_type>  SBackend;
    typedef amgcl::backend::vexcl<double>    PBackend;
#endif

    typedef
        amgcl::amg<
            PBackend,
            amgcl::runtime::coarsening::wrapper,
            amgcl::runtime::relaxation::wrapper>
        PPrecond;

    typedef
        amgcl::relaxation::as_preconditioner<
            SBackend,
            amgcl::runtime::relaxation::wrapper
            >
        SPrecond;

    typename SBackend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;
    bprm.q = ctx;

    vex::scoped_program_header header(ctx,
            amgcl::backend::vexcl_static_matrix_declaration<double, B>());
#endif

    prof.tic("setup");
    amgcl::make_solver<
        amgcl::preconditioner::cpr_drs<PPrecond, SPrecond>,
        amgcl::runtime::solver::wrapper<SBackend>
        > solve(amgcl::adapter::block_matrix<val_type>(K), prm, bprm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    size_t n = amgcl::backend::rows(K) / B;
    auto rhs_ptr = reinterpret_cast<const rhs_type*>(rhs.data());

#if defined(SOLVER_BACKEND_BUILTIN)
    auto f = amgcl::make_iterator_range(rhs_ptr, rhs_ptr + n);
#elif defined(SOLVER_BACKEND_VEXCL)
    vex::vector<rhs_type> f(ctx, n, rhs_ptr);
#endif

    auto x = SBackend::create_vector(n, bprm);
    amgcl::backend::clear(*x);

    size_t iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = solve(f, *x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
}
#endif

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using std::string;
    using std::vector;
    using amgcl::prof;
    using amgcl::precondition;

    namespace po = boost::program_options;
    namespace io = amgcl::io;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "binary,B",
         po::bool_switch()->default_value(false),
         "When specified, treat input files as binary instead of as MatrixMarket. "
         "It is assumed the files were converted to binary format with mm2bin utility. "
        )
        (
         "matrix,A",
         po::value<string>()->required(),
         "The system matrix in MatrixMarket format"
        )
        (
         "rhs,f",
         po::value<string>(),
         "The right-hand side in MatrixMarket format"
        )
        (
         "weights,w",
         po::value<string>(),
         "Equation weights in MatrixMarket format"
        )
        (
         "runtime-block-size,b",
         po::value<int>(),
         "The block size of the system matrix set at runtime"
        )
        (
         "static-block-size,c",
         po::value<int>()->default_value(1),
         "The block size of the system matrix set at compiletime"
        )
        (
         "params,P",
         po::value<string>(),
         "parameter file in json format"
        )
        (
         "prm,p",
         po::value< vector<string> >()->multitoken(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    boost::property_tree::ptree prm;
    if (vm.count("params")) read_json(vm["params"].as<string>(), prm);

    if (vm.count("prm")) {
        for(const string &v : vm["prm"].as<vector<string> >()) {
            amgcl::put(prm, v);
        }
    }

    int cb = vm["static-block-size"].as<int>();

    if (vm.count("runtime-block-size"))
        prm.put("precond.block_size", vm["runtime-block-size"].as<int>());
    else
        prm.put("precond.block_size", cb);

    size_t rows;
    vector<ptrdiff_t> ptr, col;
    vector<double> val, rhs, wgt;
    std::vector<char> pm;

    {
        auto t = prof.scoped_tic("reading");

        string Afile  = vm["matrix"].as<string>();
        bool   binary = vm["binary"].as<bool>();

        if (binary) {
            io::read_crs(Afile, rows, ptr, col, val);
        } else {
            size_t cols;
            std::tie(rows, cols) = io::mm_reader(Afile)(ptr, col, val);
            precondition(rows == cols, "Non-square system matrix");
        }

        if (vm.count("rhs")) {
            string bfile = vm["rhs"].as<string>();

            size_t n, m;

            if (binary) {
                io::read_dense(bfile, n, m, rhs);
            } else {
                std::tie(n, m) = io::mm_reader(bfile)(rhs);
            }

            precondition(n == rows && m == 1, "The RHS vector has wrong size");
        } else {
            rhs.resize(rows, 1.0);
        }

        if (vm.count("weights")) {
            string wfile = vm["weights"].as<string>();

            size_t n, m;

            if (binary) {
                io::read_dense(wfile, n, m, wgt);
            } else {
                std::tie(n, m) = io::mm_reader(wfile)(wgt);
            }

            prm.put("precond.weights",      &wgt[0]);
            prm.put("precond.weights_size", wgt.size());
        }
    }

#define CALL_BLOCK_SOLVER(z, data, B)                                          \
    case B:                                                                    \
        solve_block_cpr<B>(std::tie(rows, ptr, col, val), rhs, prm);           \
        break;

    switch(cb) {
        case 1:
            solve_cpr(std::tie(rows, ptr, col, val), rhs, prm);
            break;

#if defined(SOLVER_BACKEND_BUILTIN) || defined(SOLVER_BACKEND_VEXCL)
        BOOST_PP_SEQ_FOR_EACH(CALL_BLOCK_SOLVER, ~, AMGCL_BLOCK_SIZES)
#endif

        default:
            precondition(false, "Unsupported block size");
            break;
    }


    std::cout << prof << std::endl;
}
