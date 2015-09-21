#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> EigenMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1>          EigenVector;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1>             IntVector;

namespace amgcl {
    profiler<> prof;
} // namespace amgcl

struct pmask {
    const int *pm;

    pmask(const int *pm) : pm(pm) {}

    bool operator()(size_t i) const {
        return pm[i];
    }
};

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using amgcl::prof;
    using amgcl::precondition;

    // Read configuration from command line
    std::string parameter_file;
    std::string A_file;
    std::string pm_file;
    std::string rhs_file;
    std::string out_file = "out.mtx";

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "params,p",
         po::value<std::string>(&parameter_file),
         "parameter file in json format"
        )
        (
         "matrix,A",
         po::value<std::string>(&A_file)->required(),
         "The system matrix in MatrixMarket format"
        )
        (
         "pmask,m",
         po::value<std::string>(&pm_file)->required(),
         "The pressure mask in MatrixMarket format"
        )
        (
         "rhs,b",
         po::value<std::string>(&rhs_file),
         "The right-hand side in MatrixMarket format"
        )
        (
         "output,o",
         po::value<std::string>(&out_file),
         "The output file (saved in MatrixMarket format)"
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
    if (vm.count("params")) read_json(parameter_file, prm);

    // Read the matrix and the right-hand side.
    prof.tic("read");
    EigenMatrix A;
    precondition(
            Eigen::loadMarket(A, A_file),
            "Failed to load matrix file (" + A_file + ")"
            );

    IntVector pm;
    precondition(
            Eigen::loadMarketVector(pm, pm_file),
            "Failed to load pmask file (" + pm_file + ")"
            );

    EigenVector rhs;
    if (vm.count("rhs")) {
        precondition(
                Eigen::loadMarketVector(rhs, rhs_file),
                "Failed to load RHS file (" + rhs_file + ")"
                );
    } else {
        std::cout << "RHS was not provided; using default value of 1" << std::endl;
        rhs = EigenVector::Constant(A.rows(), 1);
    }

    precondition(A.rows() == rhs.size(), "Matrix and RHS sizes differ");
    prof.toc("read");

    // Setup CPR preconditioner
    prof.tic("setup");
    typedef
        amgcl::preconditioner::cpr<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >
        CPR;
    typedef amgcl::solver::bicgstab< amgcl::backend::builtin<double> > Solver;

    prof.tic("amg");
    amgcl::make_solver<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::ilu0,
            amgcl::solver::bicgstab
        > amg_solve( A, prm );
    prof.toc("amg");

    std::cout << amg_solve.amg() << std::endl;

    prof.tic("cpr");
    CPR cpr( A, pmask(pm.data()), prm );
    Solver solve(A.rows(), prm );
    prof.toc("cpr");
    prof.toc("setup");

    // Solve the problem
    std::vector<double> f(&rhs[0], &rhs[0] + rhs.size());
    std::vector<double> x(rhs.size(), 0);

    size_t iters;
    double resid;

    prof.tic("solve (amg)");
    boost::tie(iters, resid) = amg_solve(f, x);
    prof.toc("solve (amg)");

    std::cout << "AMG:" << std::endl
              << "  Iterations:     " << iters << std::endl
              << "  Reported Error: " << resid << std::endl
              << std::endl;

    boost::fill(x, 0);

    prof.tic("solve (cpr)");
    boost::tie(iters, resid) = solve(cpr.system_matrix(), cpr, f, x);
    prof.toc("solve (cpr)");

    std::cout << "CPR:" << std::endl
              << "  Iterations:     " << iters << std::endl
              << "  Reported Error: " << resid << std::endl
              << std::endl;

    std::cout << prof << std::endl;
}
