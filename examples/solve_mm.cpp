#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <amgcl/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> EigenMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1>          EigenVector;

namespace amgcl {
profiler<> prof;
} // namespace amgcl

//---------------------------------------------------------------------------
template<typename Matrix>
void mmread(Matrix &vec, const std::string &fname) {
    typedef typename Matrix::Scalar Scalar;

    using amgcl::precondition;

    std::ifstream in(fname.c_str());
    precondition(in, "Failed to open file \"" + fname + "\"");

    std::string line;
    int n = 0, col = 0;

    // Skip comments
    do {
        precondition(
                std::getline(in, line),
                "Format error in " + fname
                );
    } while (line[0] == '%');

    std::istringstream newline(line);
    newline >> n >> col;
    precondition(n > 0 && col > 0, "Wrong dimensions in Null-space file");
    vec.resize(n, col);

    for(int j = 0; j < col; ++j) {
        for(int i = 0; i < n; ++i) {
            Scalar v;
            precondition(
                    in >> v,
                    "Format error in " + fname
                    );

            vec(i, j) = v;
        }
    }
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using amgcl::prof;
    using amgcl::precondition;

    // Read configuration from command line
    bool just_relax = false;
    amgcl::runtime::coarsening::type coarsening = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type relaxation = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type     solver     = amgcl::runtime::solver::bicgstab;
    std::string parameter_file;
    std::string A_file;
    std::string rhs_file;
    std::string null_file;
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
         "just-relax,0",
         po::bool_switch(&just_relax),
         "Do not create AMG hierarchy, use relaxation as preconditioner"
        )
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
         "rhs,b",
         po::value<std::string>(&rhs_file),
         "The right-hand side in MatrixMarket format"
        )
        (
         "null,Z",
         po::value<std::string>(&null_file),
         "Zero energy mode vectors in MatrixMarket format"
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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Z;
    if (vm.count("null")) {
        mmread(Z, null_file);

        precondition(
                Z.rows() == A.rows(),
                "Inconsistent dimensions in Null-space file"
                );

        prm.put("precond.coarsening.nullspace.cols", Z.cols());
        prm.put("precond.coarsening.nullspace.rows", Z.rows());
        prm.put("precond.coarsening.nullspace.B",    Z.data());
    }

    precondition(A.rows() == rhs.size(), "Matrix and RHS sizes differ");
    prof.toc("read");

    std::vector<double> f(&rhs[0], &rhs[0] + rhs.size());
    std::vector<double> x(rhs.size(), 0);

    size_t iters;
    double resid;

    prm.put("solver.type", solver);

    if (just_relax) {
        std::cout << "Using relaxation as preconditioner" << std::endl;

        prm.put("precond.type", relaxation);

        prof.tic("setup");
        amgcl::make_solver<
            amgcl::runtime::relaxation::as_preconditioner<
                amgcl::backend::builtin<double>
            >,
            amgcl::runtime::iterative_solver<
                amgcl::backend::builtin<double>
            >
        > solve(A, prm);
        prof.toc("setup");

        prof.tic("solve");
        boost::tie(iters, resid) = solve(f, x);
        prof.toc("solve");
    } else {
        prm.put("precond.coarsening.type", coarsening);
        prm.put("precond.relaxation.type", relaxation);

        prof.tic("setup");
        amgcl::make_solver<
            amgcl::runtime::amg<
                amgcl::backend::builtin<double>
            >,
            amgcl::runtime::iterative_solver<
                amgcl::backend::builtin<double>
            >
        > solve(A, prm);
        prof.toc("setup");

        std::cout << solve.precond() << std::endl;

        prof.tic("solve");
        boost::tie(iters, resid) = solve(f, x);
        prof.toc("solve");
    }

    // Check the real error
    double error = (rhs - A * Eigen::Map<EigenVector>(x.data(), x.size())).norm() / rhs.norm();

    if (vm.count("out")) {
        prof.tic("write");
        Eigen::saveMarketVector(Eigen::Map<EigenVector>(x.data(), x.size()), out_file);
        prof.toc("write");
    }

    std::cout << "Iterations:     " << iters << std::endl
              << "Reported Error: " << resid << std::endl
              << "Real error:     " << error << std::endl
              << prof                        << std::endl
              ;
}
