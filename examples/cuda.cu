#include <iostream>
#include <thrust/device_vector.h>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/range/iterator_range.hpp>

#include <amgcl/backend/cuda.hpp>
#include <amgcl/relaxation/cusparse_ilu0.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl { profiler<> prof("cuda"); }
using amgcl::prof;
using amgcl::precondition;

typedef amgcl::scoped_tic< amgcl::profiler<> > scoped_tic;

//---------------------------------------------------------------------------
template <template <class> class Precond>
boost::tuple<size_t, double> solve(
        const boost::property_tree::ptree &prm,
        amgcl::backend::cuda<double>::params bprm,
        size_t rows,
        std::vector<int>    const &ptr,
        std::vector<int>    const &col,
        std::vector<double> const &val,
        std::vector<double> const &rhs,
        std::vector<double>       &x
        )
{
    typedef amgcl::backend::cuda<double> Backend;

    typedef amgcl::make_solver<
        Precond<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > Solver;

    prof.tic("setup");
    Solver solve(boost::tie(rows, ptr, col, val), prm, bprm);
    prof.toc("setup");

    std::cout << solve.precond() << std::endl;

    thrust::device_vector<double> _rhs = rhs;
    thrust::device_vector<double> _x   = x;

    boost::tuple<size_t, double> rc;

    {
        scoped_tic t(prof, "solve");
        rc = solve(_rhs, _x);
    }

    thrust::copy(_x.begin(), _x.end(), x.begin());

    return rc;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
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
    vector<int>    ptr, col;
    vector<double> val, rhs, null, x;

    if (vm.count("matrix")) {
        scoped_tic t(prof, "reading");

        using namespace amgcl::io;

        size_t cols;
        boost::tie(rows, cols) = mm_reader(vm["matrix"].as<string>())(
                ptr, col, val);

        precondition(rows == cols, "Non-square system matrix");

        if (vm.count("rhs")) {
            precondition(
                    boost::make_tuple(rows, 1) == mm_reader(vm["rhs"].as<string>())(rhs),
                    "The RHS vector has wrong size"
                    );
        } else {
            rhs.resize(rows, 1.0);
        }

        if (vm.count("null")) {
            size_t m, nv;
            boost::tie(m, nv) = mm_reader(vm["null"].as<string>())(null);
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

    amgcl::backend::cuda<double>::params bprm;
    cusparseCreate(&bprm.cusparse_handle);

    size_t iters;
    double error;

    if (vm["single-level"].as<bool>()) {
        boost::tie(iters, error) = solve<amgcl::runtime::relaxation::as_preconditioner>(
                prm, bprm, rows, ptr, col, val, rhs, x);
    } else {
        boost::tie(iters, error) = solve<amgcl::runtime::amg>(
                prm, bprm, rows, ptr, col, val, rhs, x);
    }

    if (vm.count("output")) {
        scoped_tic t(prof, "write");
        amgcl::io::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << prof << std::endl;
}

