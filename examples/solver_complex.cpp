#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/iterator_range.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/complex.hpp>

#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>

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
    typedef amgcl::backend::builtin< std::complex<double> > Backend;
    Backend::params bprm;

    typedef amgcl::make_solver<
        Precond<Backend>,
        amgcl::runtime::solver::wrapper<Backend>
        > Solver;

    prof.tic("setup");
    Solver solve(boost::tie(rows, ptr, col, val), prm, bprm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    {
        scoped_tic t(prof, "solve");
        return solve(rhs, x);
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
        scoped_tic t(prof, "reading");

        string Afile = vm["matrix"].as<string>();

        size_t cols;
        boost::tie(rows, cols) = io::mm_reader(Afile)(ptr, col, val);
        precondition(rows == cols, "Non-square system matrix");

        if (vm.count("rhs")) {
            string bfile = vm["rhs"].as<string>();

            size_t n, m;
            boost::tie(n, m) = io::mm_reader(bfile)(rhs);

            precondition(n == rows && m == 1, "The RHS vector has wrong size");
        } else {
            rhs.resize(rows, 1.0);
        }

        if (vm.count("null")) {
            string nfile = vm["null"].as<string>();

            size_t m, nv;
            boost::tie(m, nv) = io::mm_reader(nfile)(null);

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

    if (vm["single-level"].as<bool>())
        prm.put("precond.class", "relaxation");

    boost::tie(iters, error) = solve<amgcl::runtime::preconditioner>(
            prm, rows, ptr, col, val, rhs, x);

    if (vm.count("output")) {
        scoped_tic t(prof, "write");
        amgcl::io::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << prof << std::endl;
}
