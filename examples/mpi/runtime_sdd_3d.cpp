#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#if defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/backend/vexcl.hpp>
   typedef amgcl::backend::vexcl<double> Backend;
#elif defined(SOLVER_BACKEND_CUDA)
#  include <amgcl/backend/cuda.hpp>
#  include <amgcl/relaxation/cusparse_ilu0.hpp>
   typedef amgcl::backend::cuda<double> Backend;
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#  include <amgcl/backend/builtin.hpp>
   typedef amgcl::backend::builtin<double> Backend;
#endif

#include <amgcl/mpi/direct_solver/runtime.hpp>
#include <amgcl/mpi/solver/runtime.hpp>
#include <amgcl/mpi/subdomain_deflation.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

namespace amgcl {
    profiler<> prof;
}

struct deflation_vectors {
    size_t nv;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    deflation_vectors(ptrdiff_t n, size_t nv = 4) : nv(nv), x(n), y(n), z(n) {}

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
            case 3:
                return z[i];
        }
    }
};

struct renumbering {
    const domain_partition<3> &part;
    const std::vector<ptrdiff_t> &dom;

    renumbering(
            const domain_partition<3> &p,
            const std::vector<ptrdiff_t> &d
            ) : part(p), dom(d)
    {}

    ptrdiff_t operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) const {
        boost::array<ptrdiff_t, 3> p = {{i, j, k}};
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
    ptrdiff_t n = 128;
    bool constant_deflation = false;

    amgcl::runtime::coarsening::type   coarsening       = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type   relaxation       = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type       iterative_solver = amgcl::runtime::solver::bicgstabl;
    amgcl::runtime::mpi::direct::type  direct_solver    = amgcl::runtime::mpi::direct::skyline_lu;

    bool just_relax = false;
    bool symm_dirichlet = true;
    std::string parameter_file;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
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
         "gauss_seidel, ilu0, iluk, ilut, damped_jacobi, spai0, spai1, chebyshev"
        )
        (
         "iter_solver,i",
         po::value<amgcl::runtime::solver::type>(&iterative_solver)->default_value(iterative_solver),
         "cg, bicgstab, bicgstabl, gmres"
        )
        (
         "dir_solver,d",
         po::value<amgcl::runtime::mpi::direct::type>(&direct_solver)->default_value(direct_solver),
         "skyline_lu"
#ifdef AMGCL_HAVE_EIGEN
         ", eigen_splu"
#endif
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
         "params,P",
         po::value<std::string>(&parameter_file),
         "parameter file in json format"
        )
        (
         "prm,p",
         po::value< std::vector<std::string> >()->multitoken(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
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

    if (vm.count("prm")) {
        for(const std::string &v : vm["prm"].as< std::vector<std::string> >()) {
            amgcl::put(prm, v);
        }
    }

    prm.put("isolver.type", iterative_solver);
    prm.put("dsolver.type", direct_solver);

    const ptrdiff_t n3 = n * n * n;

    boost::array<ptrdiff_t, 3> lo = { {0,   0,   0  } };
    boost::array<ptrdiff_t, 3> hi = { {n-1, n-1, n-1} };

    using amgcl::prof;

    prof.tic("partition");
    domain_partition<3> part(lo, hi, world.size);
    ptrdiff_t chunk = part.size( world.rank );

    std::vector<ptrdiff_t> domain(world.size + 1);
    MPI_Allgather(
            &chunk, 1, amgcl::mpi::datatype<ptrdiff_t>(),
            &domain[1], 1, amgcl::mpi::datatype<ptrdiff_t>(), world);
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    lo = part.domain(world.rank).min_corner();
    hi = part.domain(world.rank).max_corner();

    renumbering renum(part, domain);

    deflation_vectors def(chunk, constant_deflation ? 1 : 4);
    for(ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
                boost::array<ptrdiff_t, 3> p = {{i, j, k}};
                std::pair<int,ptrdiff_t> v = part.index(p);

                def.x[v.second] = (i - (lo[0] + hi[0]) / 2);
                def.y[v.second] = (j - (lo[1] + hi[1]) / 2);
                def.z[v.second] = (k - (lo[2] + hi[2]) / 2);
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
    col.reserve(chunk * 7);
    val.reserve(chunk * 7);
    rhs.reserve(chunk);

    ptr.push_back(0);

    const double h2i  = (n - 1) * (n - 1);

    for(ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {

                if (!symm_dirichlet && (i == 0 || j == 0 || k == 0 || i + 1 == n || j + 1 == n || k + 1 == n)) {
                    col.push_back(renum(i,j,k));
                    val.push_back(1);
                    rhs.push_back(0);
                } else {
                    if (k > 0)  {
                        col.push_back(renum(i,j,k-1));
                        val.push_back(-h2i);
                    }

                    if (j > 0)  {
                        col.push_back(renum(i,j-1,k));
                        val.push_back(-h2i);
                    }

                    if (i > 0) {
                        col.push_back(renum(i-1,j,k));
                        val.push_back(-h2i);
                    }

                    col.push_back(renum(i,j,k));
                    val.push_back(6 * h2i);

                    if (i + 1 < n) {
                        col.push_back(renum(i+1,j,k));
                        val.push_back(-h2i);
                    }

                    if (j + 1 < n) {
                        col.push_back(renum(i,j+1,k));
                        val.push_back(-h2i);
                    }

                    if (k + 1 < n) {
                        col.push_back(renum(i,j,k+1));
                        val.push_back(-h2i);
                    }

                    rhs.push_back(1);
                }
                ptr.push_back( col.size() );
            }
        }
    }
    prof.toc("assemble");

    Backend::params bprm;

#if defined(SOLVER_BACKEND_VEXCL)
    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;
    bprm.q = ctx;
#elif defined(SOLVER_BACKEND_CUDA)
    cusparseCreate(&bprm.cusparse_handle);
#endif

    auto f = Backend::copy_vector(rhs, bprm);
    auto x = Backend::create_vector(chunk, bprm);

    amgcl::backend::clear(*x);

    size_t iters;
    double resid, tm_setup, tm_solve;

    std::function<double(ptrdiff_t, unsigned)> def_vec = std::cref(def);
    prm.put("num_def_vec", def.dim());
    prm.put("def_vec",     &def_vec);

    try {
    if (just_relax) {
        prm.put("local.type", relaxation);

        prof.tic("setup");
        typedef
            amgcl::mpi::subdomain_deflation<
                amgcl::relaxation::as_preconditioner<Backend, amgcl::runtime::relaxation::wrapper >,
                amgcl::runtime::mpi::solver::wrapper<Backend>,
                amgcl::runtime::mpi::direct::solver<double>
            > SDD;

        SDD solve(world, std::tie(chunk, ptr, col, val), prm, bprm);
        tm_setup = prof.toc("setup");

        prof.tic("solve");
        std::tie(iters, resid) = solve(*f, *x);
        tm_solve = prof.toc("solve");
    } else {
        prm.put("local.coarsening.type", coarsening);
        prm.put("local.relax.type", relaxation);

        prof.tic("setup");
        typedef
            amgcl::mpi::subdomain_deflation<
                amgcl::amg<Backend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper>,
                amgcl::runtime::mpi::solver::wrapper<Backend>,
                amgcl::runtime::mpi::direct::solver<double>
            > SDD;

        SDD solve(world, std::tie(chunk, ptr, col, val), prm, bprm);
        tm_setup = prof.toc("setup");

        prof.tic("solve");
        std::tie(iters, resid) = solve(*f, *x);
        tm_solve = prof.toc("solve");
    }
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        throw e;
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
        log_name << "log3d_" << n3 << "_" << nt << "_" << world.size << ".txt";
        std::ofstream log(log_name.str().c_str(), std::ios::app);
        log << n3 << "\t" << nt << "\t" << world.size
            << "\t" << tm_setup << "\t" << tm_solve
            << "\t" << iters << "\t" << std::endl;
    }

}
