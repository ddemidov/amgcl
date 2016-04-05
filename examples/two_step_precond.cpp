#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/make_solver.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/preconditioner/simple.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
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
    std::string parameter_file;
    std::string A_file;
    std::string pm_file;
    std::string rhs_file;
    std::string out_file = "out.mtx";
    amgcl::runtime::coarsening::type coarsening = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type prelax     = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::relaxation::type frelax     = amgcl::runtime::relaxation::ilu0;
    amgcl::runtime::solver::type     solver     = amgcl::runtime::solver::bicgstab;

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
         "The pressure mask in MatrixMarket format. Or, if the parameter has "
         "the form '%n:m', then each (n+i*m)-th variable is treated as pressure."
        )
        (
         "rhs,b",
         po::value<std::string>(&rhs_file),
         "The right-hand side in MatrixMarket format"
        )
        (
         "coarsening,c",
         po::value<amgcl::runtime::coarsening::type>(&coarsening)->default_value(coarsening),
         "ruge_stuben, aggregation, smoothed_aggregation, smoothed_aggr_emin"
        )
        (
         "pressure_relaxation,r",
         po::value<amgcl::runtime::relaxation::type>(&prelax)->default_value(prelax),
         "gauss_seidel, multicolor_gauss_seidel, ilu0, damped_jacobi, spai0, chebyshev"
        )
        (
         "flow_relaxation,f",
         po::value<amgcl::runtime::relaxation::type>(&frelax)->default_value(frelax),
         "gauss_seidel, multicolor_gauss_seidel, ilu0, damped_jacobi, spai0, chebyshev"
        )
        (
         "solver,s",
         po::value<amgcl::runtime::solver::type>(&solver)->default_value(solver),
         "cg, bicgstab, bicgstabl, gmres"
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

    // Read the matrix and the right-hand side.
    prof.tic("read");
    size_t rows, cols;
    std::vector<int> ptr, col;
    std::vector<double> val;
    boost::tie(rows, cols) = amgcl::io::mm_reader(A_file)(ptr, col, val);

    std::vector<char> pm;
    if (pm_file[0] == '%') {
        int start  = std::atoi(pm_file.substr(1).c_str());
        int stride = std::atoi(pm_file.substr(3).c_str());
        pm.resize(rows, 0);
        for(size_t i = start; i < rows; i += stride) pm[i] = 1;
    } else {
        size_t n, m;
        boost::tie(n, m) = amgcl::io::mm_reader(pm_file)(pm);
    }

    std::vector<double> rhs;
    if (vm.count("rhs")) {
        size_t n, m;
        boost::tie(n, m) = amgcl::io::mm_reader(rhs_file)(rhs);
    } else {
        std::cout << "RHS was not provided; using default value of 1" << std::endl;
        rhs.resize(rows, 1.0);
    }

    precondition(rows == rhs.size(), "Matrix and RHS sizes differ");

    boost::property_tree::ptree prm;
    if (vm.count("params")) read_json(parameter_file, prm);

    prm.put("precond.pressure.coarsening.type", coarsening);
    prm.put("precond.pressure.relaxation.type", prelax);
    prm.put("precond.flow.type", frelax);
    prm.put("precond.pmask", static_cast<void*>(&pm[0]));
    prm.put("precond.pmask_size", pm.size());
    prm.put("solver.type", solver);

    prof.toc("read");

    // Setup CPR preconditioner
    prof.tic("setup");
    prof.tic("cpr");
    amgcl::make_solver<
        amgcl::preconditioner::cpr<
            amgcl::runtime::amg<
                amgcl::backend::builtin<double>
                >,
            amgcl::runtime::relaxation::as_preconditioner<
                amgcl::backend::builtin<double>
                >
            >,
        amgcl::runtime::iterative_solver<
            amgcl::backend::builtin<double>
            >
        > cpr( boost::tie(rows, ptr, col, val), prm );
    prof.toc("cpr");

    prof.tic("simple");
    amgcl::make_solver<
        amgcl::preconditioner::simple<
            amgcl::runtime::amg<
                amgcl::backend::builtin<double>
                >,
            amgcl::runtime::relaxation::as_preconditioner<
                amgcl::backend::builtin<double>
                >
            >,
        amgcl::runtime::iterative_solver<
            amgcl::backend::builtin<double>
            >
        > simple( boost::tie(rows, ptr, col, val), prm );
    prof.toc("simple");
    prof.toc("setup");

    // Solve the problem
    std::vector<double> x(rows);

    size_t iters;
    double resid;

    prof.tic("solve");
    boost::fill(x, 0);

    prof.tic("cpr");
    boost::tie(iters, resid) = cpr(rhs, x);
    prof.toc("cpr");

    std::cout << "CPR:" << std::endl
              << "  Iterations:     " << iters << std::endl
              << "  Reported Error: " << resid << std::endl
              << std::endl;

    boost::fill(x, 0);

    prof.tic("simple");
    boost::tie(iters, resid) = simple(rhs, x);
    prof.toc("simple");

    std::cout << "SIMPLE:" << std::endl
              << "  Iterations:     " << iters << std::endl
              << "  Reported Error: " << resid << std::endl
              << std::endl;
    prof.toc("solve");

    std::cout << prof << std::endl;
}
