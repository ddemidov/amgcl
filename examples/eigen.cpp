#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>
#include <boost/range/iterator_range.hpp>

#include <amgcl/backend/eigen.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> Matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1>          Vector;

// Adapt AMGCL preconditioner to Eigen
template <class Precond>
struct eigen_adapter {
    const Precond &P;

    eigen_adapter(const Precond &P) : P(P) {}

    Vector solve(const Vector &f) const {
        Vector x = Vector::Zero(f.size());
        P.apply(f, x);
        return x;
    }
};

template<class Precond>
eigen_adapter<Precond> adapt(const Precond &P) {
    return eigen_adapter<Precond>(P);
}

//---------------------------------------------------------------------------
template <template <class> class Precond>
void solve(
        const boost::property_tree::ptree &prm,
        const Matrix &A, const Vector &f,
        Vector &x
        )
{
    prof.tic("setup");
    Precond< amgcl::backend::eigen<double> > P(A, prm);
    std::cout << P << std::endl;
    prof.toc("setup");

    using Eigen::internal::bicgstab;

    int    iters = 100;
    double error = 1e-2;

    prof.tic("solve");
    bicgstab(A, f, x, adapt(P), iters, error);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
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
         "May be provided multiple times. Example:\n"
         "  -p relax.type=ilu0"
        )
        ("matrix,A",
         po::value<string>()->required(),
         "System matrix in the MatrixMarket format. "
        )
        (
         "rhs,f",
         po::value<string>(),
         "The RHS vector in the MatrixMarket format. "
         "When omitted, a vector of ones is used by default. "
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

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

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

    Matrix A;
    Vector f;

    precondition(Eigen::loadMarket(A, vm["matrix"].as<string>()),
            "Failed to load matrix file");

    if (vm.count("rhs")) {
        precondition(Eigen::loadMarketVector(f, vm["rhs"].as<string>()),
                "Failed to load RHS file");
    } else {
        f = Vector::Constant(A.rows(), 1);
    }

    precondition(A.rows() == f.size(), "Matrix and RHS sizes differ");
    Vector x = Vector::Constant(f.size(), vm.count("x") ? vm["x"].as<double>() : 0.0);

    if (vm["single-level"].as<bool>()) {
        solve<amgcl::runtime::relaxation::as_preconditioner>(prm, A, f, x);
    } else {
        solve<amgcl::runtime::amg>(prm, A, f, x);
    }

    if (vm.count("output")) {
        amgcl::io::mm_write(vm["output"].as<string>(), &x[0], x.size());
    }

    std::cout << prof << std::endl;
}
