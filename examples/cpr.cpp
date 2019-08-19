#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>
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
    auto t1 = prof.scoped_tic("CPR");

    typedef amgcl::backend::builtin<double> Backend;

    typedef
        amgcl::amg<Backend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper>
        PPrecond;

    typedef
        amgcl::relaxation::as_preconditioner<Backend, amgcl::runtime::relaxation::wrapper>
        SPrecond;

    amgcl::make_solver<
        amgcl::preconditioner::cpr<PPrecond, SPrecond>,
        amgcl::runtime::solver::wrapper<Backend>
        > solve(K, prm);

    std::cout << solve.precond() << std::endl;

    auto t2 = prof.scoped_tic("solve");
    std::vector<double> x(rhs.size(), 0.0);

    size_t iters;
    double error;

    std::tie(iters, error) = solve(rhs, x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
}

//---------------------------------------------------------------------------
template <int B, class Matrix>
void solve_block_cpr(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    auto t1 = prof.scoped_tic("CPR");

    typedef amgcl::static_matrix<double, B, B> val_type;
    typedef amgcl::static_matrix<double, B, 1> rhs_type;
    typedef amgcl::backend::builtin<val_type>  SBackend;
    typedef amgcl::backend::builtin<double>    PBackend;

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

    amgcl::make_solver<
        amgcl::preconditioner::cpr<PPrecond, SPrecond>,
        amgcl::runtime::solver::wrapper<SBackend>
        > solve(amgcl::adapter::block_matrix<val_type>(K), prm);

    std::cout << solve.precond() << std::endl;

    auto t2 = prof.scoped_tic("solve");
    std::vector<rhs_type> x(rhs.size(), amgcl::math::zero<rhs_type>());

    auto rhs_ptr = reinterpret_cast<const rhs_type*>(rhs.data());
    size_t n = amgcl::backend::rows(K) / B;

    size_t iters;
    double error;

    std::tie(iters, error) = solve(amgcl::make_iterator_range(rhs_ptr, rhs_ptr + n), x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
}

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
    }

#define CALL_BLOCK_SOLVER(z, data, B)                                          \
    case B:                                                                    \
        solve_block_cpr<B>(std::tie(rows, ptr, col, val), rhs, prm);           \
        break;

    switch(cb) {
        case 1:
            solve_cpr(std::tie(rows, ptr, col, val), rhs, prm);
            break;

        BOOST_PP_SEQ_FOR_EACH(CALL_BLOCK_SOLVER, ~, AMGCL_BLOCK_SIZES)

        default:
            precondition(false, "Unsupported block size");
            break;
    }

    std::cout << prof << std::endl;
}
