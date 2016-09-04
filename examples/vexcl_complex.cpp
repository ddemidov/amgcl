#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>
#include <boost/range/iterator_range.hpp>

#include <amgcl/value_type/complex.hpp>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_complex.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/zero_copy.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

typedef amgcl::scoped_tic< amgcl::profiler<> > scoped_tic;

//---------------------------------------------------------------------------
template <template <class> class Precond>
boost::tuple<size_t, double> solve(
        const boost::property_tree::ptree &prm,
        size_t rows,
        std::vector<ptrdiff_t> const &ptr,
        std::vector<ptrdiff_t> const &col,
        std::vector< std::complex<double> > const &val,
        std::vector< std::complex<double> > const &rhs,
        std::vector< std::complex<double> >       &x
        )
{
    typedef amgcl::backend::vexcl< std::complex<double> > Backend;

    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;

    amgcl::backend::enable_complex_for_vexcl(ctx);

    typename Backend::params bprm;
    bprm.q = ctx;

    typedef amgcl::make_solver<
        Precond<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > Solver;

    prof.tic("setup");
    Solver solve(amgcl::adapter::zero_copy(rows, &ptr[0], &col[0], &val[0]), prm, bprm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    scoped_tic t(prof, "solve");

    vex::vector< std::complex<double> > f(ctx, rhs);
    vex::vector< std::complex<double> > u(ctx, x);

    boost::tuple<size_t, double> r = solve(f, u);
    vex::copy(u, x);
    return r;
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
         po::value< vector<string> >(),
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

    size_t rows;
    vector<ptrdiff_t> ptr, col;
    vector< std::complex<double> > val, rhs, null, x;

    if (vm.count("matrix")) {
        scoped_tic t(prof, "reading");

        string Afile  = vm["matrix"].as<string>();
        bool   binary = vm["binary"].as<bool>();

        if (binary) {
            io::read_crs(Afile, rows, ptr, col, val);
        } else {
            size_t cols;
            boost::tie(rows, cols) = io::mm_reader(Afile)(ptr, col, val);
            precondition(rows == cols, "Non-square system matrix");
        }

        if (vm.count("rhs")) {
            string bfile = vm["rhs"].as<string>();

            size_t n, m;

            if (binary) {
                io::read_dense(bfile, n, m, rhs);
            } else {
                boost::tie(n, m) = io::mm_reader(bfile)(rhs);
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
                boost::tie(m, nv) = io::mm_reader(nfile)(null);
            }

            precondition(m == rows, "Near null-space vectors have wrong size");

            prm.put("precond.coarsening.nullspace.cols", nv);
            prm.put("precond.coarsening.nullspace.rows", rows);
            prm.put("precond.coarsening.nullspace.B",    &null[0]);
        }
    } else {
        scoped_tic t(prof, "assembling");
        rows = sample_problem(vm["size"].as<int>(), val, col, ptr, rhs);
    }

    x.resize(rows, vm["initial"].as<double>());

    size_t iters;
    double error;

    bool single_level = vm["single-level"].as<bool>();

    if (single_level) {
        boost::tie(iters, error) = solve<amgcl::runtime::relaxation::as_preconditioner>(
                prm, rows, ptr, col, val, rhs, x);
    } else {
        boost::tie(iters, error) = solve<amgcl::runtime::amg>(
                prm, rows, ptr, col, val, rhs, x);
    }

    double norm_rhs = sqrt(std::abs(amgcl::backend::inner_product(rhs, rhs)));
    amgcl::backend::spmv(-1.0, boost::tie(rows, ptr, col, val), x, 1.0, rhs);
    double resid = sqrt(std::abs(amgcl::backend::inner_product(rhs, rhs))) / norm_rhs;

    if (vm.count("output")) {
        scoped_tic t(prof, "write");
        amgcl::io::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }

    std::cout << "Iterations:     " << iters << std::endl
              << "Reported error: " << error << std::endl
              << "Real error:     " << resid << std::endl
              << prof << std::endl;
}
