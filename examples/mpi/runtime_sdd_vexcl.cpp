#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/mpi/direct_solver.hpp>
#include <amgcl/mpi/subdomain_deflation.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

//#define RECIRCULATION

namespace amgcl {
    profiler<> prof;
}

struct linear_deflation {
    std::vector<double> x;
    std::vector<double> y;

    linear_deflation(ptrdiff_t n) : x(n), y(n) {}

    size_t dim() const { return 3; }

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

    amgcl::runtime::coarsening::type   coarsening       = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type   relaxation       = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type       iterative_solver = amgcl::runtime::solver::bicgstabl;
    amgcl::runtime::mpi::dsolver::type direct_solver    = amgcl::runtime::mpi::dsolver::skyline_lu;

    std::string problem = "laplace2d";
    std::string parameter_file;
    bool just_relax = false;

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
         "gauss_seidel, ilu0, iluk, ilut, damped_jacobi, spai0, spai1, chebyshev"
        )
        (
         "iter_solver,i",
         po::value<amgcl::runtime::solver::type>(&iterative_solver)->default_value(iterative_solver),
         "cg, bicgstab, bicgstabl, gmres"
        )
        (
         "dir_solver,d",
         po::value<amgcl::runtime::mpi::dsolver::type>(&direct_solver)->default_value(direct_solver),
         "skyline_lu"
#ifdef AMGCL_HAVE_PASTIX
         ", pastix"
#endif
        )
        (
         "params,p",
         po::value<std::string>(&parameter_file),
         "parameter file in json format"
        )
        (
         "just-relax,0",
         po::bool_switch(&just_relax),
         "Do not create AMG hierarchy, use relaxation as preconditioner"
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

    prm.put("isolver.type", iterative_solver);
    prm.put("dsolver.type", direct_solver);

    const ptrdiff_t n2 = n * n;

    boost::array<ptrdiff_t, 2> lo = { {0, 0} };
    boost::array<ptrdiff_t, 2> hi = { {n - 1, n - 1} };

    using amgcl::prof;

    prof.tic("partition");
    domain_partition<2> part(lo, hi, world.size);
    ptrdiff_t chunk = part.size( world.rank );

    std::vector<ptrdiff_t> domain(world.size + 1);
    MPI_Allgather(
            &chunk, 1, amgcl::mpi::datatype<ptrdiff_t>(),
            &domain[1], 1, amgcl::mpi::datatype<ptrdiff_t>(),
            world
            );
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    ptrdiff_t chunk_start = domain[world.rank];
    ptrdiff_t chunk_end   = domain[world.rank + 1];

    linear_deflation lindef(chunk);
    std::vector<ptrdiff_t> renum(n2);
    for(ptrdiff_t j = 0, idx = 0; j < n; ++j) {
        for(ptrdiff_t i = 0; i < n; ++i, ++idx) {
            boost::array<ptrdiff_t, 2> p = {{i, j}};
            std::pair<int,ptrdiff_t> v = part.index(p);
            renum[idx] = domain[v.first] + v.second;

            boost::array<ptrdiff_t,2> lo = part.domain(v.first).min_corner();
            boost::array<ptrdiff_t,2> hi = part.domain(v.first).max_corner();

            if (v.first == world.rank) {
                lindef.x[v.second] = (i - (lo[0] + hi[0]) / 2);
                lindef.y[v.second] = (j - (lo[1] + hi[1]) / 2);
            }
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

    const double hinv = (n - 1);
    const double h2i  = (n - 1) * (n - 1);

    if (problem == "recirc2d") {
        const double h    = 1 / hinv;
        const double eps  = 1e-5;

        for(ptrdiff_t j = 0, idx = 0; j < n; ++j) {
            double y = h * j;
            for(ptrdiff_t i = 0; i < n; ++i, ++idx) {
                double x = h * i;

                if (renum[idx] < chunk_start || renum[idx] >= chunk_end) continue;

                if (i == 0 || j == 0 || i + 1 == n || j + 1 == n) {
                    col.push_back(renum[idx]);
                    val.push_back(1);
                    rhs.push_back(
                            sin(M_PI * x) + sin(M_PI * y) +
                            sin(13 * M_PI * x) + sin(13 * M_PI * y)
                            );
                } else {
                    double a = -sin(M_PI * x) * cos(M_PI * y) * hinv;
                    double b =  sin(M_PI * y) * cos(M_PI * x) * hinv;

                    if (j > 0) {
                        col.push_back(renum[idx - n]);
                        val.push_back(-eps * h2i - std::max(b, 0.0));
                    }

                    if (i > 0) {
                        col.push_back(renum[idx - 1]);
                        val.push_back(-eps * h2i - std::max(a, 0.0));
                    }

                    col.push_back(renum[idx]);
                    val.push_back(4 * eps * h2i + fabs(a) + fabs(b));

                    if (i + 1 < n) {
                        col.push_back(renum[idx + 1]);
                        val.push_back(-eps * h2i + std::min(a, 0.0));
                    }

                    if (j + 1 < n) {
                        col.push_back(renum[idx + n]);
                        val.push_back(-eps * h2i + std::min(b, 0.0));
                    }

                    rhs.push_back(1.0);
                }
                ptr.push_back( col.size() );
            }
        }
    } else {
        for(ptrdiff_t j = 0, idx = 0; j < n; ++j) {
            for(ptrdiff_t i = 0; i < n; ++i, ++idx) {
                if (renum[idx] < chunk_start || renum[idx] >= chunk_end) continue;

                if (j > 0)  {
                    col.push_back(renum[idx - n]);
                    val.push_back(-h2i);
                }

                if (i > 0) {
                    col.push_back(renum[idx - 1]);
                    val.push_back(-h2i - hinv);
                }

                col.push_back(renum[idx]);
                val.push_back(4 * h2i + hinv);

                if (i + 1 < n) {
                    col.push_back(renum[idx + 1]);
                    val.push_back(-h2i);
                }

                if (j + 1 < n) {
                    col.push_back(renum[idx + n]);
                    val.push_back(-h2i);
                }

                rhs.push_back(1);
                ptr.push_back( col.size() );
            }
        }
    }
    prof.toc("assemble");

    vex::Context ctx(
            vex::Filter::Env &&
            vex::Filter::DoublePrecision &&
            vex::Filter::Count(1)
            );

    typedef amgcl::backend::vexcl<double> Backend;
    Backend::params bprm;
    bprm.q = ctx.queue();

    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(ctx, chunk);
    x = 0;

    size_t iters;
    double resid, tm_setup, tm_solve;

    boost::function<double(ptrdiff_t, unsigned)> def_vec = boost::cref(lindef);
    prm.put("num_def_vec", lindef.dim());
    prm.put("def_vec",     &def_vec);

    if (just_relax) {
        prm.put("local.type", relaxation);

        prof.tic("setup");
        typedef
            amgcl::mpi::subdomain_deflation<
                amgcl::runtime::relaxation::as_preconditioner< amgcl::backend::vexcl<double> >,
                amgcl::runtime::iterative_solver,
                amgcl::runtime::mpi::direct_solver<double>
            > SDD;

        SDD solve(world, boost::tie(chunk, ptr, col, val), prm, bprm);
        tm_setup = prof.toc("setup");

        prof.tic("solve");
        boost::tie(iters, resid) = solve(f, x);
        tm_solve = prof.toc("solve");
    } else {
        prm.put("local.coarsening.type", coarsening);
        prm.put("local.relax.type", relaxation);

        prof.tic("setup");
        typedef
            amgcl::mpi::subdomain_deflation<
                amgcl::runtime::amg< amgcl::backend::vexcl<double> >,
                amgcl::runtime::iterative_solver,
                amgcl::runtime::mpi::direct_solver<double>
            > SDD;

        SDD solve(world, boost::tie(chunk, ptr, col, val), prm, bprm);
        tm_setup = prof.toc("setup");

        prof.tic("solve");
        boost::tie(iters, resid) = solve(f, x);
        tm_solve = prof.toc("solve");
    }

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
