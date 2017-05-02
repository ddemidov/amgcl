#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>

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
solve_schur(const Matrix&, const std::vector<double>&, boost::property_tree::ptree&)
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

    tic t1(prof, "schur_complement");

    amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<USolver, PSolver>,
        amgcl::runtime::iterative_solver<Backend>
        > solve(K, prm, bprm);

    std::cout << solve.precond() << std::endl;

    typedef Backend::vector vector;
    boost::shared_ptr<vector> f = Backend::copy_vector(rhs, bprm);
    boost::shared_ptr<vector> x = Backend::create_vector(rhs.size(), bprm);
    amgcl::backend::clear(*x);

    tic t2(prof, "solve");
    size_t iters;
    double error;

    boost::tie(iters, error) = solve(*f, *x);

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
                typedef
                    amgcl::make_solver<
                        amgcl::runtime::amg<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
#if defined(SOLVER_BACKEND_BUILTIN)
        case 2:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > BBackend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::amg<BBackend>,
                        amgcl::runtime::iterative_solver<BBackend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
        case 3:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 3, 3> > BBackend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::amg<BBackend>,
                        amgcl::runtime::iterative_solver<BBackend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
        case 4:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 4, 4> > BBackend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::amg<BBackend>,
                        amgcl::runtime::iterative_solver<BBackend>
                        >
                    PSolver;
                solve_schur<USolver, PSolver>(K, rhs, prm);
            }
            break;
#endif
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
                typedef
                    amgcl::make_solver<
                        amgcl::runtime::relaxation::as_preconditioner<Backend>,
                        amgcl::runtime::iterative_solver<Backend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
#if defined(SOLVER_BACKEND_BUILTIN)
        case 2:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 2, 2> > BBackend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::relaxation::as_preconditioner<BBackend>,
                        amgcl::runtime::iterative_solver<BBackend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
        case 3:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 3, 3> > BBackend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::relaxation::as_preconditioner<BBackend>,
                        amgcl::runtime::iterative_solver<BBackend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
        case 4:
            {
                typedef amgcl::backend::builtin< amgcl::static_matrix<double, 4, 4> > BBackend;
                typedef
                    amgcl::make_block_solver<
                        amgcl::runtime::relaxation::as_preconditioner<BBackend>,
                        amgcl::runtime::iterative_solver<BBackend>
                        >
                    USolver;
                solve_schur<USolver>(pb, K, rhs, prm);
            }
            break;
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
