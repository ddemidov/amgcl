#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#if defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/backend/vexcl.hpp>
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
#  include <amgcl/value_type/static_matrix.hpp>
#  include <amgcl/adapter/block_matrix.hpp>
#  include <amgcl/make_block_solver.hpp>
   typedef amgcl::backend::builtin<double> Backend;
#endif

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

#ifndef AMGCL_BLOCK_SIZES
#  define AMGCL_BLOCK_SIZES (3)(4)
#endif

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
template <class USolver, class PSolver, class Matrix>
void solve_schur(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
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

    auto t1 = prof.scoped_tic("schur_complement");

    prof.tic("setup");
    amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<USolver, PSolver>,
        amgcl::runtime::solver::wrapper<Backend>
        > solve(K, prm, bprm);
    prof.toc("setup");

    std::cout << solve << std::endl;

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

#define AMGCL_BLOCK_PSOLVER(z, data, B)                                        \
  case B: {                                                                    \
    typedef amgcl::backend::builtin<amgcl::static_matrix<double, B, B> >       \
        BBackend;                                                              \
    typedef amgcl::make_block_solver<                                          \
        amgcl::runtime::preconditioner<BBackend>,                              \
        amgcl::runtime::solver::wrapper<BBackend> >                            \
        PSolver;                                                               \
    solve_schur<USolver, PSolver>(K, rhs, prm);                                \
  } break;

//---------------------------------------------------------------------------
template <class USolver, class Matrix>
void solve_schur(int pb, const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    switch (pb) {
        case 1:
            {
                typedef
                    amgcl::make_solver<
                        amgcl::runtime::preconditioner<Backend>,
                        amgcl::runtime::solver::wrapper<Backend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
#if defined(SOLVER_BACKEND_BUILTIN)
        BOOST_PP_SEQ_FOR_EACH(AMGCL_BLOCK_PSOLVER, ~, AMGCL_BLOCK_SIZES)
#endif
        default:
            precondition(false, "Unsupported block size for pressure");
    }
}

#define AMGCL_BLOCK_USOLVER(z, data, B)                                        \
  case B: {                                                                    \
    typedef amgcl::backend::builtin<amgcl::static_matrix<double, B, B> >       \
        BBackend;                                                              \
    typedef amgcl::make_block_solver<                                          \
        amgcl::runtime::preconditioner<BBackend>,                              \
        amgcl::runtime::solver::wrapper<BBackend> >                            \
        USolver;                                                               \
    solve_schur<USolver>(pb, K, rhs, prm);                                     \
  } break;

//---------------------------------------------------------------------------
template <class Matrix>
void solve_schur(int ub, int pb, const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    switch (ub) {
        case 1:
            {
                typedef
                    amgcl::make_solver<
                        amgcl::runtime::preconditioner<Backend>,
                        amgcl::runtime::solver::wrapper<Backend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
#if defined(SOLVER_BACKEND_BUILTIN)
        BOOST_PP_SEQ_FOR_EACH(AMGCL_BLOCK_USOLVER, ~, AMGCL_BLOCK_SIZES)
#endif
        default:
            precondition(false, "Unsupported block size for flow");
    }
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using std::string;
    using std::vector;

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
         "pmask,m",
         po::value<string>(),
         "The pressure mask in MatrixMarket format. Or, if the parameter has "
         "the form '%n:m', then each (n+i*m)-th variable is treated as pressure."
        )
        (
         "ub",
         po::value<int>()->default_value(1),
         "Block-size of the 'flow'/'non-pressure' part of the matrix"
        )
        (
         "pb",
         po::value<int>()->default_value(1),
         "Block-size of the 'pressure' part of the matrix"
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

    size_t rows;
    vector<ptrdiff_t> ptr, col;
    vector<double> val, rhs;
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

        if(vm.count("pmask")) {
            std::string pmask = vm["pmask"].as<string>();
            prm.put("precond.pmask_size", rows);

            switch (pmask[0]) {
                case '%':
                case '<':
                case '>':
                    prm.put("precond.pmask_pattern", pmask);
                    break;
                default:
                    {
                        size_t n, m;
                        std::tie(n, m) = amgcl::io::mm_reader(pmask)(pm);
                        precondition(n == rows && m == 1, "Mask file has wrong size");
                        prm.put("precond.pmask", static_cast<void*>(&pm[0]));
                    }
            }
        }
    }

    solve_schur(vm["ub"].as<int>(), vm["pb"].as<int>(),
            std::tie(rows, ptr, col, val), rhs, prm);

    std::cout << prof << std::endl;
}
