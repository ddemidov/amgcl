#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#if defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/backend/vexcl.hpp>
#  include <amgcl/backend/vexcl_static_matrix.hpp>
   template <class T> using Backend = amgcl::backend::vexcl<T>;
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#  include <amgcl/backend/builtin.hpp>
#  include <amgcl/value_type/static_matrix.hpp>
#  include <amgcl/adapter/block_matrix.hpp>
#  include <amgcl/make_block_solver.hpp>
   template <class T> using Backend = amgcl::backend::builtin<T>;
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
    typedef Backend<double> SBackend;
    SBackend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;
    bprm.q = ctx;

    typedef typename amgcl::math::scalar_of<typename USolver::backend_type::value_type>::type u_scalar;
    typedef typename amgcl::math::scalar_of<typename USolver::backend_type::value_type>::type p_scalar;

    const int UB = amgcl::math::static_rows<typename USolver::backend_type::value_type>::value;
    const int PB = amgcl::math::static_rows<typename USolver::backend_type::value_type>::value;

    std::list<vex::scoped_program_header> headers;
    if (UB > 1) headers.emplace_back(ctx, amgcl::backend::vexcl_static_matrix_declaration<float,UB>());
    if (PB > 1) headers.emplace_back(ctx, amgcl::backend::vexcl_static_matrix_declaration<float,PB>());
#endif

    auto t1 = prof.scoped_tic("schur_complement");

    prof.tic("setup");
    amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<USolver, PSolver>,
        amgcl::runtime::solver::wrapper<SBackend>
        > solve(K, prm, bprm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    auto f = SBackend::copy_vector(rhs, bprm);
    auto x = SBackend::create_vector(rhs.size(), bprm);
    amgcl::backend::clear(*x);

    size_t iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = solve(K, *f, *x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
}

#define AMGCL_BLOCK_PSOLVER(z, data, B)                                        \
  case B: {                                                                    \
    typedef Backend<amgcl::static_matrix<float, B, B>> BBackend;               \
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
                        amgcl::runtime::preconditioner<Backend<float>>,
                        amgcl::runtime::solver::wrapper<Backend<float>>
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
    typedef Backend<amgcl::static_matrix<float, B, B>> BBackend;               \
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
                        amgcl::runtime::preconditioner<Backend<float>>,
                        amgcl::runtime::solver::wrapper<Backend<float>>
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
         "scale,s",
         po::bool_switch()->default_value(false),
         "Scale the matrix so that the diagonal is unit. "
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

                        if (binary) {
                            io::read_dense(pmask, n, m, pm);
                        } else {
                            std::tie(n, m) = amgcl::io::mm_reader(pmask)(pm);
                        }

                        precondition(n == rows && m == 1, "Mask file has wrong size");

                        prm.put("precond.pmask", static_cast<void*>(&pm[0]));
                    }
            }
        }
    }

    if (vm["scale"].as<bool>()) {
        std::vector<double> dia(rows, 1.0);

        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(rows); ++i) {
            double d = 1.0;
            for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
                if (col[j] == i) {
                    d = 1 / sqrt(val[j]);
                }
            }
            if (!std::isnan(d)) dia[i] = d;
        }

        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(rows); ++i) {
            rhs[i] *= dia[i];
            for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
                val[j] *= dia[i] * dia[col[j]];
            }
        }
    }

    solve_schur(vm["ub"].as<int>(), vm["pb"].as<int>(),
            std::tie(rows, ptr, col, val), rhs, prm);

    std::cout << prof << std::endl;
}
