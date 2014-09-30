#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <amgcl/runtime.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> EigenMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1>          EigenVector;

namespace amgcl {
profiler<> prof;

namespace backend {
template <> struct is_builtin_vector<EigenVector> : boost::true_type {};

} // namespace backend
} // namespace amgcl

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using amgcl::prof;

    // Read configuration from command line
    amgcl::runtime::coarsening::type coarsening = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type relaxation = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type     solver     = amgcl::runtime::solver::bicgstab;
    std::string parameter_file;
    std::string A_file   = "A.mtx";
    std::string rhs_file = "rhs.mtx";
    std::string out_file = "out.mtx";

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "coarsening,c",
         po::value<amgcl::runtime::coarsening::type>(&coarsening)->default_value(coarsening),
         "ruge_stuben, aggregation, smoothed_aggregation, smoothed_aggr_emin"
        )
        (
         "relaxation,r",
         po::value<amgcl::runtime::relaxation::type>(&relaxation)->default_value(relaxation),
         "gauss_seidel, ilu0, damped_jacobi, spai0, chebyshev"
        )
        (
         "solver,s",
         po::value<amgcl::runtime::solver::type>(&solver)->default_value(solver),
         "cg, bicgstab, bicgstabl, gmres"
        )
        (
         "params,p",
         po::value<std::string>(&parameter_file),
         "parameter file in json format"
        )
        (
         "matrix,A",
         po::value<std::string>(&A_file)->default_value(A_file),
         "The system matrix in MatrixMarket format"
        )
        (
         "rhs,b",
         po::value<std::string>(&rhs_file)->default_value(rhs_file),
         "The right-hand side in MatrixMarket format"
        )
        (
         "output,o",
         po::value<std::string>(&out_file)->default_value(out_file),
         "The output file (saved in MatrixMarket format)"
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
    if (vm.count("params")) read_json(parameter_file, prm);

    // Read the matrix and the right-hand side.
    prof.tic("read");
    EigenMatrix A;
    amgcl::precondition(
            Eigen::loadMarket(A, A_file),
            "Failed to load matrix file (" + A_file + ")"
            );

    EigenVector rhs;
    amgcl::precondition(
            Eigen::loadMarketVector(rhs, rhs_file),
            "Failed to load RHS file (" + rhs_file + ")"
            );

    amgcl::precondition(A.rows() == rhs.size(), "Matrix and RHS sizes differ");
    prof.toc("read");

    // Setup solver
    prof.tic("setup");
    typedef
        amgcl::runtime::make_solver< amgcl::backend::builtin<double> >
        Solver;

    Solver solve(coarsening, relaxation, solver, A, prm);
    prof.toc("setup");

    std::cout << solve.amg() << std::endl;

    // Solve the problem
    EigenVector x = EigenVector::Zero(rhs.size());

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(rhs, x);
    prof.toc("solve");

    prof.tic("write");
    Eigen::saveMarketVector(x, out_file);
    prof.toc("write");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl      << prof  << std::endl;

}
