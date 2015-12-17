#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/scope_exit.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/range/algorithm.hpp>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <boost/multi_array.hpp>

#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/mpi/runtime.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

namespace amgcl {
    profiler<> prof;
}

struct deflation {
    virtual size_t dim() const = 0;
    virtual double operator()(ptrdiff_t i, unsigned j) const = 0;
    virtual ~deflation() {}
};

struct constant_deflation : public deflation {
    size_t dim() const { return 1; }
    double operator()(ptrdiff_t i, unsigned j) const { return 1.0; }
};

struct linear_deflation : public deflation {
    std::vector<double> x;
    std::vector<double> y;

    linear_deflation(
            ptrdiff_t chunk,
            boost::array<ptrdiff_t, 2> lo,
            boost::array<ptrdiff_t, 2> hi
            )
    {
        double hx = 1.0 / (hi[0] - lo[0]);
        double hy = 1.0 / (hi[1] - lo[1]);

        ptrdiff_t nx = hi[0] - lo[0] + 1;
        ptrdiff_t ny = hi[1] - lo[1] + 1;

        x.reserve(chunk);
        y.reserve(chunk);

        for (ptrdiff_t j = 0; j < ny; ++j) {
            for (ptrdiff_t i = 0; i < nx; ++i) {
                x.push_back(i * hx - 0.5);
                y.push_back(j * hy - 0.5);
            }
        }
    }

    size_t dim() const { return 3; }

    double operator()(ptrdiff_t i, unsigned j) const {
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

struct bilinear_deflation : public deflation {
    size_t nv, chunk;
    std::vector<double> v;

    bilinear_deflation(
            ptrdiff_t n,
            ptrdiff_t chunk,
            boost::array<ptrdiff_t, 2> lo,
            boost::array<ptrdiff_t, 2> hi
            ) : chunk(chunk)
    {
        // See which neighbors we have.
        int neib[2][2] = {
            {lo[0] > 0 || lo[1] > 0,     hi[0] + 1 < n || lo[1] > 0    },
            {lo[0] > 0 || hi[1] + 1 < n, hi[0] + 1 < n || hi[1] + 1 < n}
        };

        for(int j = 0; j < 2; ++j)
            for(int i = 0; i < 2; ++i)
                if (neib[j][i]) ++nv;

        if (nv == 0) {
            // Single MPI process?
            nv = 1;
            v.resize(chunk, 1);
            return;
        }

        v.resize(chunk * nv, 0);

        double *dv = v.data();

        ptrdiff_t nx = hi[0] - lo[0] + 1;
        ptrdiff_t ny = hi[1] - lo[1] + 1;

        double hx = 1.0 / (nx - 1);
        double hy = 1.0 / (ny - 1);

        for(int j = 0; j < 2; ++j) {
            for(int i = 0; i < 2; ++i) {
                if (!neib[j][i]) continue;

                boost::multi_array_ref<double, 2> V(dv, boost::extents[ny][nx]);

                for(ptrdiff_t jj = 0; jj < ny; ++jj) {
                    double y = jj * hy;
                    double b = std::abs((1 - j) - y);
                    for(ptrdiff_t ii = 0; ii < nx; ++ii) {
                        double x = ii * hx;

                        double a = std::abs((1 - i) - x);
                        V[jj][ii] = a * b;
                    }
                }

                dv += chunk;
            }
        }
    }

    size_t dim() const { return nv; }

    double operator()(ptrdiff_t i, unsigned j) const {
        return v[j * chunk + i];
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
    std::string deflation_type = "bilinear";

    amgcl::runtime::coarsening::type    coarsening       = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type    relaxation       = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type        iterative_solver = amgcl::runtime::solver::bicgstabl;
    amgcl::runtime::direct_solver::type direct_solver    = amgcl::runtime::direct_solver::skyline_lu;

    bool just_relax = false;
    bool symm_dirichlet = true;
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
         "deflation,v",
         po::value<std::string>(&deflation_type)->default_value(deflation_type),
         "Deflation type (constant, linear, bilinear)"
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
    prof.toc("partition");

    renumbering renum(part, domain);

    prof.tic("deflation");
    boost::shared_ptr<deflation> def;

    if (deflation_type == "constant")
        def.reset(new constant_deflation());
    else if (deflation_type == "linear")
        def.reset(new linear_deflation(chunk, lo, hi));
    else if (deflation_type == "bilinear")
        def.reset(new bilinear_deflation(n, chunk, lo, hi));
    else
        throw std::runtime_error("Unsupported deflation type");
    prof.toc("deflation");

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

    std::vector<double> x(chunk, 0);
    size_t iters;
    double resid, tm_setup, tm_solve;

    if (just_relax) {
        prm.put("precond.type", relaxation);

        prof.tic("setup");
        typedef
            amgcl::runtime::mpi::subdomain_deflation<
                amgcl::runtime::relaxation::as_preconditioner< amgcl::backend::builtin<double> >
                >
            SDD;

        SDD solve(world, boost::tie(chunk, ptr, col, val), *def, prm);
        tm_setup = prof.toc("setup");

        prof.tic("solve");
        boost::tie(iters, resid) = solve(rhs, x);
        tm_solve = prof.toc("solve");
    } else {
        prm.put("precond.coarsening.type", coarsening);
        prm.put("precond.relaxation.type", relaxation);

        prof.tic("setup");
        typedef
            amgcl::runtime::mpi::subdomain_deflation<
                amgcl::runtime::amg< amgcl::backend::builtin<double> >
                >
            SDD;

        SDD solve(world, boost::tie(chunk, ptr, col, val), *def, prm);
        tm_setup = prof.toc("setup");

        prof.tic("solve");
        boost::tie(iters, resid) = solve(rhs, x);
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
