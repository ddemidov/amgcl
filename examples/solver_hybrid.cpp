#include <iostream>
#include <string>
#include <random>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/preprocessor/seq/for_each.hpp>


#if defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/value_type/static_matrix.hpp>
#  include <amgcl/adapter/block_matrix.hpp>
#  include <amgcl/backend/vexcl.hpp>
#  include <amgcl/backend/vexcl_static_matrix.hpp>
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#  include <amgcl/backend/builtin.hpp>
#  include <amgcl/value_type/static_matrix.hpp>
#endif

#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/coarsening/rigid_body_modes.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/reorder.hpp>
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
template <int B>
std::tuple<size_t, double> solve(
        const boost::property_tree::ptree &prm,
        size_t rows,
        std::vector<ptrdiff_t> const &ptr,
        std::vector<ptrdiff_t> const &col,
        std::vector<double>    const &val,
        std::vector<double>    const &rhs,
        std::vector<double>          &x
        )
{
    typedef amgcl::static_matrix<double, B, B> block_type;
#if defined(SOLVER_BACKEND_VEXCL)
    typedef amgcl::backend::vexcl_hybrid<double, block_type> Backend;
#else
    typedef amgcl::backend::builtin_hybrid<double, block_type> Backend;
#endif

    typedef amgcl::make_solver<
        amgcl::runtime::preconditioner<Backend>,
        amgcl::runtime::solver::wrapper<Backend>
        > Solver;

    typename Backend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;
    bprm.q = ctx;

    vex::scoped_program_header header(ctx,
            amgcl::backend::vexcl_static_matrix_declaration<double,B>());
#endif

    auto A = std::tie(rows, ptr, col, val);

    prof.tic("setup");
    Solver solve(A, prm, bprm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    auto f_b = Backend::copy_vector(rhs, bprm);
    auto x_b = Backend::copy_vector(x,   bprm);

    prof.tic("solve");
    auto info = solve(*f_b, *x_b);
    prof.toc("solve");

#if defined(SOLVER_BACKEND_VEXCL)
    vex::copy(*x_b, x);
#else
    std::copy(&(*x_b)[0], &(*x_b)[0] + rows, &x[0]);
#endif

    return info;
}

#define AMGCL_CALL_SOLVER(z, data, B)                                          \
  case B:                                                                      \
    return solve<B>(prm, rows, ptr, col, val, rhs, x);

//---------------------------------------------------------------------------
std::tuple<size_t, double> solve(
        const boost::property_tree::ptree &prm,
        size_t rows,
        std::vector<ptrdiff_t> const &ptr,
        std::vector<ptrdiff_t> const &col,
        std::vector<double>    const &val,
        std::vector<double>    const &rhs,
        std::vector<double>          &x,
        int block_size
        )
{
    switch (block_size) {
        BOOST_PP_SEQ_FOR_EACH(AMGCL_CALL_SOLVER, ~, AMGCL_BLOCK_SIZES)
        default:
            precondition(false, "Unsupported block size");
            return std::make_tuple(0, 0.0);
    }
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    namespace io = amgcl::io;

    using amgcl::prof;
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
         po::value< vector<string> >()->multitoken(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        ("matrix,A",
         po::value<string>()->required(),
         "System matrix in the MatrixMarket format. "
         "When not specified, solves a Poisson problem in 3D unit cube. "
        )
        (
         "rhs,f",
         po::value<string>()->required(),
         "The RHS vector in the MatrixMarket format. "
         "When omitted, a vector of ones is used by default. "
         "Should only be provided together with a system matrix. "
        )
        (
         "null,N",
         po::value<string>(),
         "The near null-space vectors in the MatrixMarket format. "
         "Should be a dense matrix of size N*M, where N is the number of "
         "unknowns, and M is the number of null-space vectors. "
         "Should only be provided together with a system matrix. "
        )
        (
         "coords,C",
         po::value<string>(),
         "Coordinate matrix where number of rows corresponds to the number of grid nodes "
         "and the number of columns corresponds to the problem dimensionality (2 or 3). "
         "Will be used to construct near null-space vectors as rigid body modes. "
         "Should only be provided together with a system matrix. "
        )
        (
         "binary,B",
         po::bool_switch()->default_value(false),
         "When specified, treat input files as binary instead of as MatrixMarket. "
         "It is assumed the files were converted to binary format with mm2bin utility. "
        )
        (
         "block-size,b",
         po::value<int>()->required(),
         "The block size of the system matrix. "
         "When specified, the system matrix is assumed to have block-wise structure. "
         "This usually is the case for problems in elasticity, structural mechanics, "
         "for coupled systems of PDE (such as Navier-Stokes equations), etc. "
        )
        (
         "single-level,1",
         po::bool_switch()->default_value(false),
         "When specified, the AMG hierarchy is not constructed. "
         "Instead, the problem is solved using a single-level smoother as preconditioner. "
        )
        (
         "initial,x",
         po::value<double>()->default_value(0),
         "Value to use as initial approximation. "
        )
        (
         "random-initial",
         po::bool_switch()->default_value(false),
         "Use random initial approximation. "
        )
        (
         "output,o",
         po::value<string>(),
         "Output file. Will be saved in the MatrixMarket format. "
         "When omitted, the solution is not saved. "
        )
        ;

    po::positional_options_description p;
    p.add("prm", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    for (int i = 0; i < argc; ++i) {
        if (i) std::cout << " ";
        std::cout << argv[i];
    }
    std::cout << std::endl;

    boost::property_tree::ptree prm;
    if (vm.count("prm-file")) {
        read_json(vm["prm-file"].as<string>(), prm);
    }

    if (vm.count("prm")) {
        for(const string &v : vm["prm"].as<vector<string> >()) {
            amgcl::put(prm, v);
        }
    }

    size_t rows, nv = 0;
    vector<ptrdiff_t> ptr, col;
    vector<double> val, rhs, null, x;

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

        string bfile = vm["rhs"].as<string>();

        size_t n, m;

        if (binary) {
            io::read_dense(bfile, n, m, rhs);
        } else {
            std::tie(n, m) = io::mm_reader(bfile)(rhs);
        }

        precondition(n == rows && m == 1, "The RHS vector has wrong size");

        if (vm.count("null")) {
            string nfile = vm["null"].as<string>();

            size_t m;

            if (binary) {
                io::read_dense(nfile, m, nv, null);
            } else {
                std::tie(m, nv) = io::mm_reader(nfile)(null);
            }

            precondition(m == rows, "Near null-space vectors have wrong size");
        } else if (vm.count("coords")) {
            string cfile = vm["coords"].as<string>();
            std::vector<double> coo;

            size_t m, ndim;

            if (binary) {
                io::read_dense(cfile, m, ndim, coo);
            } else {
                std::tie(m, ndim) = io::mm_reader(cfile)(coo);
            }

            precondition(m * ndim == rows && (ndim == 2 || ndim == 3), "Coordinate matrix has wrong size");

            nv = amgcl::coarsening::rigid_body_modes(ndim, coo, null);
        }

        if (nv) {
            prm.put("precond.coarsening.nullspace.cols", nv);
            prm.put("precond.coarsening.nullspace.rows", rows);
            prm.put("precond.coarsening.nullspace.B",    &null[0]);
        }
    }

    x.resize(rows, vm["initial"].as<double>());
    if (vm["random-initial"].as<bool>()) {
        std::mt19937 rng;
        std::uniform_real_distribution<double> rnd(-1, 1);
        for(auto &v : x) v = rnd(rng);
    }

    size_t iters;
    double error;

    int block_size = vm["block-size"].as<int>();

    if (vm["single-level"].as<bool>())
        prm.put("precond.class", "relaxation");

    std::tie(iters, error) = solve(prm, rows, ptr, col, val, rhs, x, block_size);

    if (vm.count("output")) {
        auto t = prof.scoped_tic("write");
        amgcl::io::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << prof << std::endl;
}
