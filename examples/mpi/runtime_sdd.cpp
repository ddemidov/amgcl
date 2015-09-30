#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/scope_exit.hpp>
#include <boost/range/algorithm.hpp>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/mpi/runtime.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

namespace amgcl {
    profiler<> prof;
}

struct deflation_vectors {
    size_t nv;
    std::vector<double> x;
    std::vector<double> y;

    deflation_vectors(ptrdiff_t n, size_t nv = 3) : nv(nv), x(n), y(n) {}

    size_t dim() const { return nv; }

    double operator()(ptrdiff_t i, int j) const {
        switch(j) {
            default:
            case 0:
                return 1;
            case 1:
                return x[i];
            case 2:
                return y[i];
        }
    }
};

struct renumbering {
    const domain_partition<2> &part;
    const std::vector<ptrdiff_t> &dom;

    renumbering(
            const domain_partition<2> &p,
            const std::vector<ptrdiff_t> &d
            ) : part(p), dom(d)
    {}

    ptrdiff_t operator()(ptrdiff_t i, ptrdiff_t j) const {
        boost::array<ptrdiff_t, 2> p = {{i, j}};
        std::pair<int,ptrdiff_t> v = part.index(p);
        return dom[v.first] + v.second;
    }
};

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    // Read configuration from command line
    ptrdiff_t n = 1024;
    bool constant_deflation = false;

    amgcl::runtime::coarsening::type    coarsening       = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type    relaxation       = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type        iterative_solver = amgcl::runtime::solver::bicgstabl;
    amgcl::runtime::direct_solver::type direct_solver    = amgcl::runtime::direct_solver::skyline_lu;

    bool        symm_dirichlet = true;
    std::string problem = "laplace2d";
    std::string parameter_file;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "problem",
         po::value<std::string>(&problem)->default_value(problem),
         "laplace2d, recirc2d"
        )
        (
         "symbc",
         po::value<bool>(&symm_dirichlet)->default_value(symm_dirichlet),
         "Use symmetric Dirichlet conditions in laplace2d"
        )
        (
         "size,n",
         po::value<ptrdiff_t>(&n)->default_value(n),
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
         "gauss_seidel, multicolor_gauss_seidel, ilu0, damped_jacobi, spai0, chebyshev"
        )
        (
         "iter_solver,i",
         po::value<amgcl::runtime::solver::type>(&iterative_solver)->default_value(iterative_solver),
         "cg, bicgstab, bicgstabl, gmres"
        )
        (
         "dir_solver,d",
         po::value<amgcl::runtime::direct_solver::type>(&direct_solver)->default_value(direct_solver),
         "skyline_lu"
#ifdef AMGCL_HAVE_PASTIX
         ", pastix"
#endif
        )
        (
         "cd",
         po::bool_switch(&constant_deflation),
         "Use constant deflation (linear deflation is used by default)"
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

    prm.put("amg.coarsening.type", coarsening);
    prm.put("amg.relaxation.type", relaxation);
    prm.put("solver.type",         iterative_solver);
    prm.put("direct_solver.type",  direct_solver);

    const ptrdiff_t n2 = n * n;
    const double hinv  = (n - 1);
    const double h2i   = (n - 1) * (n - 1);
    const double h     = 1 / hinv;


    boost::array<ptrdiff_t, 2> lo = { {0, 0} };
    boost::array<ptrdiff_t, 2> hi = { {n - 1, n - 1} };

    using amgcl::prof;

    prof.tic("partition");
    domain_partition<2> part(lo, hi, world.size);
    ptrdiff_t chunk = part.size( world.rank );

    std::vector<ptrdiff_t> domain(world.size + 1);
    MPI_Allgather(
            &chunk, 1, amgcl::mpi::datatype<ptrdiff_t>::get(),
            &domain[1], 1, amgcl::mpi::datatype<ptrdiff_t>::get(), world);
    boost::partial_sum(domain, domain.begin());

    lo = part.domain(world.rank).min_corner();
    hi = part.domain(world.rank).max_corner();

    renumbering renum(part, domain);

    deflation_vectors def(chunk, constant_deflation ? 1 : 3);
    for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
        for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
            boost::array<ptrdiff_t, 2> p = {{i, j}};
            std::pair<int,ptrdiff_t> v = part.index(p);

            def.x[v.second] = h * (i - (lo[0] + hi[0]) / 2);
            def.y[v.second] = h * (j - (lo[1] + hi[1]) / 2);
        }
    }
    prof.toc("partition");

    prof.tic("assemble");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 5);
    val.reserve(chunk * 5);
    rhs.reserve(chunk);

    ptr.push_back(0);

    if (problem == "recirc2d") {
        const double eps  = 1e-5;

        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            double y = h * j;
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
                double x = h * i;

                if (i == 0 || j == 0 || i + 1 == n || j + 1 == n) {
                    col.push_back(renum(i,j));
                    val.push_back(1);
                    rhs.push_back(
                            sin(M_PI * x) + sin(M_PI * y) +
                            sin(13 * M_PI * x) + sin(13 * M_PI * y)
                            );
                } else {
                    double a = -sin(M_PI * x) * cos(M_PI * y) * hinv;
                    double b =  sin(M_PI * y) * cos(M_PI * x) * hinv;

                    if (j > 0) {
                        col.push_back(renum(i,j-1));
                        val.push_back(-eps * h2i - std::max(b, 0.0));
                    }

                    if (i > 0) {
                        col.push_back(renum(i-1,j));
                        val.push_back(-eps * h2i - std::max(a, 0.0));
                    }

                    col.push_back(renum(i,j));
                    val.push_back(4 * eps * h2i + fabs(a) + fabs(b));

                    if (i + 1 < n) {
                        col.push_back(renum(i+1,j));
                        val.push_back(-eps * h2i + std::min(a, 0.0));
                    }

                    if (j + 1 < n) {
                        col.push_back(renum(i,j+1));
                        val.push_back(-eps * h2i + std::min(b, 0.0));
                    }

                    rhs.push_back(1.0);
                }
                ptr.push_back( col.size() );
            }
        }
    } else {
        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
                if (!symm_dirichlet && (i == 0 || j == 0 || i + 1 == n || j + 1 == n)) {
                    col.push_back(renum(i,j));
                    val.push_back(1);
                    rhs.push_back(0);
                } else {
                    if (j > 0)  {
                        col.push_back(renum(i,j-1));
                        val.push_back(-h2i);
                    }

                    if (i > 0) {
                        col.push_back(renum(i-1,j));
                        val.push_back(-h2i);
                    }

                    col.push_back(renum(i,j));
                    val.push_back(4 * h2i);

                    if (i + 1 < n) {
                        col.push_back(renum(i+1,j));
                        val.push_back(-h2i);
                    }

                    if (j + 1 < n) {
                        col.push_back(renum(i,j+1));
                        val.push_back(-h2i);
                    }

                    rhs.push_back(1);
                }
                ptr.push_back( col.size() );
            }
        }
    }
    prof.toc("assemble");

    prof.tic("setup");
    typedef
        amgcl::runtime::mpi::subdomain_deflation<
            amgcl::backend::builtin<double>
            >
        SDD;

    SDD solve(world, boost::tie(chunk, ptr, col, val), def, prm);
    double tm_setup = prof.toc("setup");

    if (world.rank == 0) {
        boost::property_tree::ptree actual_params;
        solve.get_params(actual_params);
        write_json(std::cout, actual_params);
    }

    std::vector<double> x(chunk, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(rhs, x);
    double tm_solve = prof.toc("solve");

    if (world.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << resid << std::endl
            << std::endl
            << prof << std::endl;

#ifdef _OPENMP
        int nt = omp_get_max_threads();
#else
        int nt = 1;
#endif
        std::ostringstream log_name;
        log_name << "log_" << n2 << "_" << nt << "_" << world.size << ".txt";
        std::ofstream log(log_name.str().c_str(), std::ios::app);
        log << n2 << "\t" << nt << "\t" << world.size
            << "\t" << tm_setup << "\t" << tm_solve
            << "\t" << iters << "\t" << std::endl;
    }

}
