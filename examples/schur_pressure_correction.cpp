#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>

#include <amgcl/make_solver.hpp>
#include <amgcl/make_block_solver.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

typedef amgcl::scoped_tic< amgcl::profiler<> > tic;

template <class USolver, class PSolver, class Matrix>
typename boost::enable_if_c<
    (
        amgcl::math::static_rows<
            typename amgcl::preconditioner::detail::common_backend<
                typename USolver::backend_type,
                typename PSolver::backend_type
            >::type::value_type
        >::value > 1
    ),
    void
    >::type
solve_schur(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{}

//---------------------------------------------------------------------------
template <class USolver, class PSolver, class Matrix>
typename boost::enable_if_c<
    (
        amgcl::math::static_rows<
            typename amgcl::preconditioner::detail::common_backend<
                typename USolver::backend_type,
                typename PSolver::backend_type
            >::type::value_type
        >::value == 1
    ),
    void>::type
solve_schur(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    tic t1(prof, "schur_complement");

    typedef amgcl::backend::builtin<double> Backend;

    amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<USolver, PSolver>,
        amgcl::runtime::iterative_solver<Backend>
        > solve(K, prm);

    std::cout << solve.precond() << std::endl;

    tic t2(prof, "solve");
    std::vector<double> x(rhs.size(), 0.0);

    size_t iters;
    double error;

    boost::tie(iters, error) = solve(rhs, x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
}

//---------------------------------------------------------------------------
template <class USolver, class Matrix>
void solve_schur(int pb, const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    switch (pb) {
        case 1:
            {
                typedef amgcl::backend::builtin<double> Backend;
                typedef
                    amgcl::make_solver<
                        amgcl::runtime::amg<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
        case 2:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > Backend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::amg<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
        case 3:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > Backend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::amg<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
        case 4:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > Backend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::amg<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
        default:
            precondition(false, "Unsupported block size for pressure");
    }
}

//---------------------------------------------------------------------------
template <class Matrix>
void solve_schur(int ub, int pb, const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    precondition(ub == 1 || pb == 1,
            "At least one of the flow/pressure subproblems has to be scalar");

    switch (ub) {
        case 1:
            {
                typedef amgcl::backend::builtin<double> Backend;
                typedef
                    amgcl::make_solver<
                        amgcl::runtime::relaxation::as_preconditioner<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
        case 2:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > Backend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::relaxation::as_preconditioner<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
        case 3:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > Backend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::relaxation::as_preconditioner<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
        case 4:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > Backend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::relaxation::as_preconditioner<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
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
         po::value<string>()->required(),
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
         po::value< vector<string> >(),
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
    vector<double> val, rhs;
    std::vector<char> pm;

    {
        tic t(prof, "reading");

        string Afile  = vm["matrix"].as<string>();
        bool   binary = vm["binary"].as<bool>();
        string mfile  = vm["pmask"].as<string>();

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

        if (mfile[0] == '%') {
            int start  = std::atoi(mfile.substr(1).c_str());
            int stride = std::atoi(mfile.substr(3).c_str());
            pm.resize(rows, 0);
            for(size_t i = start; i < rows; i += stride) pm[i] = 1;
        } else {
            size_t n, m;

            if (binary) {
                io::read_dense(mfile, n, m, pm);
            } else {
                boost::tie(n, m) = amgcl::io::mm_reader(mfile)(pm);
            }

            precondition(n == rows && m == 1, "Mask file has wrong size");
        }
    }

    prm.put("precond.pmask", static_cast<void*>(&pm[0]));
    prm.put("precond.pmask_size", pm.size());

    solve_schur(vm["ub"].as<int>(), vm["pb"].as<int>(),
            boost::tie(rows, ptr, col, val), rhs, prm);

    std::cout << prof << std::endl;
}
