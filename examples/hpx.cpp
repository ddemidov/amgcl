#include <iostream>
#include <hpx/hpx_init.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/hpx.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    amgcl::profiler<> prof;
}

//---------------------------------------------------------------------------
int hpx_main(boost::program_options::variables_map &vm) {
    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;

    using amgcl::prof;

    prof.tic("assemble");
    int n = sample_problem(vm["size"].as<int>(), val, col, ptr, rhs);
    prof.toc("assemble");

    prof.tic("setup");
    typedef amgcl::backend::HPX<double> Backend;

    typedef amgcl::make_solver<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0,
        amgcl::solver::bicgstab
        > Solver;

    Solver::params prm;
    prm.amg.backend.grain_size = vm["grain"].as<int>();

    Solver solve( boost::tie(n, ptr, col, val), prm );
    prof.toc("setup");

    std::cout << solve.amg() << std::endl;

    auto f = Backend::copy_vector(rhs, prm.amg.backend);
    auto x = Backend::create_vector(n, prm.amg.backend);

    amgcl::backend::clear(*x);

    int    iters;
    double error;
    prof.tic("solve");
    hpx::reset_active_counters();
    boost::tie(iters, error) = solve(*f, *x);
    prof.toc("solve");
    std::cout
        << "Iters: " << iters << std::endl
        << "Error: " << error << std::endl
        << prof               << std::endl
        ;

    return hpx::finalize();
}
//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    boost::program_options::options_description
       options("Usage: " HPX_APPLICATION_STRING " [options]");

    options.add_options()
        (
         "size,n",
         boost::program_options::value<int>()->default_value(32),
         "problem size"
        )
        (
         "grain,g",
         boost::program_options::value<int>()->default_value(4096),
         "grain size"
        )
        ;
    return hpx::init(options, argc, argv);
}
