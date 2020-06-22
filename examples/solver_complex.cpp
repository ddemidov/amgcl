#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/complex.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>

#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>

#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

#ifndef AMGCL_BLOCK_SIZES
#  define AMGCL_BLOCK_SIZES (2)(3)(4)
#endif

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
template <class Precond, class Matrix>
std::tuple<size_t, double> solve(
        const Matrix &A,
        const boost::property_tree::ptree &prm,
        std::vector< std::complex<double> > const &f,
        std::vector< std::complex<double> >       &x
        )
{
    typedef typename Precond::backend_type Backend;

    typedef typename amgcl::math::rhs_of<typename Backend::value_type>::type rhs_type;
    size_t n = amgcl::backend::rows(A);

    rhs_type const * fptr = reinterpret_cast<rhs_type const *>(&f[0]);
    rhs_type       * xptr = reinterpret_cast<rhs_type       *>(&x[0]);
    amgcl::iterator_range<rhs_type const *> frng(fptr, fptr + n);
    amgcl::iterator_range<rhs_type       *> xrng(xptr, xptr + n);

    typedef amgcl::make_solver<
        Precond,
        amgcl::runtime::solver::wrapper<Backend>
        > Solver;

    prof.tic("setup");
    Solver solve(A, prm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    {
        auto t = prof.scoped_tic("solve");
        return solve(frng, xrng);
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
         po::value<string>(),
         "System matrix in the MatrixMarket format. "
         "When not specified, solves a Poisson problem in 3D unit cube. "
        )
        (
         "rhs,f",
         po::value<string>(),
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
         "binary,B",
         po::bool_switch()->default_value(false),
         "When specified, treat input files as binary instead of as MatrixMarket. "
         "It is assumed the files were converted to binary format with mm2bin utility. "
        )
        (
         "block-size,b",
         po::value<int>()->default_value(1),
         "The block size of the system matrix. "
         "When specified, the system matrix is assumed to have block-wise structure. "
         "This usually is the case for problems in elasticity, structural mechanics, "
         "for coupled systems of PDE (such as Navier-Stokes equations), etc. "
        )
        (
         "size,n",
         po::value<int>()->default_value(32),
         "The size of the Poisson problem to solve when no system matrix is given. "
         "Specified as number of grid nodes along each dimension of a unit cube. "
         "The resulting system will have n*n*n unknowns. "
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
         "output,o",
         po::value<string>(),
         "Output file. Will be saved in the MatrixMarket format. "
         "When omitted, the solution is not saved. "
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
        for(const string &v : vm["prm"].as<vector<string> >()) {
            amgcl::put(prm, v);
        }
    }

    size_t rows;
    vector<ptrdiff_t> ptr, col;
    vector< std::complex<double> > val, rhs, null, x;

    if (vm.count("matrix")) {
        auto t = prof.scoped_tic("reading");

        string Afile = vm["matrix"].as<string>();
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

        if (vm.count("null")) {
            string nfile = vm["null"].as<string>();

            size_t m, nv;

            if (binary) {
                io::read_dense(nfile, m, nv, null);
            } else {
                std::tie(m, nv) = io::mm_reader(nfile)(null);
            }

            precondition(m == rows, "Near null-space vectors have wrong size");

            prm.put("precond.coarsening.nullspace.cols", nv);
            prm.put("precond.coarsening.nullspace.rows", rows);
            prm.put("precond.coarsening.nullspace.B",    &null[0]);
        }
    } else {
        auto t = prof.scoped_tic("assembling");
        rows = sample_problem(vm["size"].as<int>(), val, col, ptr, rhs);
    }

    x.resize(rows, vm["initial"].as<double>());

    size_t iters;
    double error;

    if (vm["single-level"].as<bool>())
        prm.put("precond.class", "relaxation");

    int block_size = vm["block-size"].as<int>();

#define CALL_BLOCK_SOLVER(z, data, B)                                                    \
        case B:                                                                          \
            {                                                                            \
                typedef amgcl::static_matrix<std::complex<double>,B,B> value_type;       \
                typedef amgcl::backend::builtin<value_type> Backend;                     \
                std::tie(iters, error) = solve<amgcl::runtime::preconditioner<Backend>>( \
                        amgcl::adapter::block_matrix<value_type>(                        \
                            std::tie(rows, ptr, col, val)),                              \
                        prm, rhs, x);                                                    \
            }                                                                            \
            break;

    switch (block_size) {
        case 1:
            {
                typedef amgcl::backend::builtin<std::complex<double>> Backend;
                std::tie(iters, error) = solve<amgcl::runtime::preconditioner<Backend>>(
                        std::tie(rows, ptr, col, val), prm, rhs, x);
            }
            break;
        BOOST_PP_SEQ_FOR_EACH(CALL_BLOCK_SOLVER, ~, AMGCL_BLOCK_SIZES)
    }

#undef CALL_BLOCK_SOLVER

    if (vm.count("output")) {
        auto t = prof.scoped_tic("write");
        amgcl::io::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << prof << std::endl;
}
