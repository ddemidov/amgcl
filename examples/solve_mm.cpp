#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/adapter/zero_copy.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
profiler<> prof;
} // namespace amgcl

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
         "gauss_seidel, ilu0, parallel_ilu0, damped_jacobi, spai0, chebyshev"
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
    size_t rows, cols;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val;
    boost::tie(rows, cols) = amgcl::io::mm_reader(A_file)(ptr, col, val);

    std::vector<double> rhs;
    if (vm.count("rhs")) {
        size_t n, m;
        boost::tie(n, m) = amgcl::io::mm_reader(rhs_file)(rhs);
    } else {
        std::cout << "RHS was not provided; using default value of 1" << std::endl;
        rhs.resize(rows, 1.0);
    }

    std::vector<double> Z;
    if (vm.count("null")) {
        size_t Zrows, Zcols;
        boost::tie(Zrows, Zcols) = amgcl::io::mm_reader(null_file)(Z);

        precondition(
                Zrows == rows,
                "Inconsistent dimensions in Null-space file"
                );

        prm.put("precond.coarsening.nullspace.cols", Zcols);
        prm.put("precond.coarsening.nullspace.rows", Zrows);
        prm.put("precond.coarsening.nullspace.B",    &Z[0]);
    }

    precondition(rows == rhs.size(), "Matrix and RHS sizes differ");
    prof.toc("read");

    std::vector<double> x(rows, 0);

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
        > solve(amgcl::adapter::zero_copy(rows, &ptr[0], &col[0], &val[0]), prm);
        prof.toc("setup");

        prof.tic("solve");
        boost::tie(iters, resid) = solve(rhs, x);
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
        > solve(amgcl::adapter::zero_copy(rows, &ptr[0], &col[0], &val[0]), prm);
        prof.toc("setup");

        std::cout << solve.precond() << std::endl;

        prof.tic("solve");
        boost::tie(iters, resid) = solve(rhs, x);
        prof.toc("solve");
    }

    if (vm.count("out")) {
        prof.tic("write");
        amgcl::io::mm_write(out_file, &x[0], x.size());
        prof.toc("write");
    }

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << prof << std::endl
              ;
}
