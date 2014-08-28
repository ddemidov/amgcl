#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/runtime.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

int main(int argc, char *argv[]) {
    using amgcl::prof;

    // Read configuration from command line
    int m = 32;
    amgcl::runtime::coarsening::type coarsening = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type relaxation = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type     solver     = amgcl::runtime::solver::bicgstab;
    std::string parameter_file;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "size,n",
         po::value<int>(&m)->default_value(m),
         "domain size"
        )
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

    // Assemble problem
    prof.tic("assemble");
    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    int n = sample_problem(m, val, col, ptr, rhs);
    prof.toc("assemble");

    typedef
        amgcl::runtime::make_solver< amgcl::backend::builtin<double> >
        Solver;

    // Setup solver
    prof.tic("setup");
    Solver solve(
            coarsening, relaxation, solver,
            boost::tie(n, ptr, col, val), prm
            );
    prof.toc("setup");

    std::cout << solve.amg() << std::endl;

    // Solve the problem
    std::vector<double> x(n, 0);

    size_t iters;
    double resid;
    prof.tic("solve");
    boost::tie(iters, resid) = solve(rhs, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl      << prof  << std::endl;
}
