#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof;
}

struct dummy {
    template <class Vec1, class Vec2>
    void apply(const Vec1 &rhs, Vec2 &x) const {
        std::copy(rhs.begin(), rhs.end(), x.begin());
    }
};

int main(int argc, char *argv[]) {
    using amgcl::prof;

    // Read configuration from command line
    int m = 32;
    double x0 = 0;
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
         "params,p",
         po::value<std::string>(&parameter_file),
         "parameter file in json format"
        )
        (
         "x0",
         po::value<double>(&x0),
         "Initial approximation value"
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
        amgcl::relaxation::as_preconditioner<
            amgcl::backend::builtin<double>,
            amgcl::relaxation::ilu0
            >
        Precond;

    typedef
        amgcl::solver::cg<
            amgcl::backend::builtin<double>
            >
        Solver;

    // Setup the solution
    prof.tic("setup");
    Precond P(boost::tie(n, ptr, col, val), prm, prm);
    Solver solve(n, prm);
    prof.toc("setup");

    // Solve the problem
    std::vector<double> x(n, x0);

    size_t iters;
    double resid;
    prof.tic("solve");
    boost::tie(iters, resid) = solve(P.system_matrix(), P, rhs, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl      << prof  << std::endl;
}
