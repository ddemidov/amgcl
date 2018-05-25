#include <iostream>
#include <vector>
#include <string>

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/mpi/direct_solver/runtime.hpp>
#include <amgcl/mpi/coarsening/smoothed_aggregation.hpp>
#include <amgcl/mpi/coarsening/smoothed_pmis.hpp>
#include <amgcl/mpi/relaxation/runtime.hpp>

#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

namespace amgcl {
    profiler<> prof;
}

namespace math = amgcl::math;

//typedef amgcl::static_matrix<double, 2, 2>       val_type;
typedef double val_type;

typedef typename math::rhs_of<val_type>::type    rhs_type;
typedef typename math::scalar_of<val_type>::type scalar;

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

template <template <class> class Coarsening>
void solve(
    const std::vector<ptrdiff_t> &ptr,
    const std::vector<ptrdiff_t> &col,
    const std::vector<val_type>  &val,
    const boost::property_tree::ptree &prm
    )
{
    typedef amgcl::backend::builtin<val_type> Backend;
    typedef
        amgcl::mpi::make_solver<
            amgcl::mpi::amg<
                Backend, Coarsening<Backend>,
                amgcl::runtime::mpi::relaxation<Backend>,
                amgcl::runtime::mpi::direct::solver<val_type>
                >,
            amgcl::runtime::solver::wrapper
            >
        Solver;

    using amgcl::prof;

    amgcl::mpi::communicator comm(MPI_COMM_WORLD);

    size_t n = ptr.size() - 1;

    prof.tic("setup");
    Solver solve(comm, boost::tie(n, ptr, col, val), prm);
    prof.toc("setup");

    if (comm.rank == 0) {
        std::cout << solve << std::endl;
    }

    std::vector<rhs_type> f(n, math::constant<rhs_type>(1)), x(n, math::zero<rhs_type>());
    int    iters;
    double error;

    prof.tic("solve");
    boost::tie(iters, error) = solve(f, x);
    prof.toc("solve");

    if (comm.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << error << std::endl
            << prof << std::endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator comm(MPI_COMM_WORLD);

    if (comm.rank == 0)
        std::cout << "World size: " << comm.size << std::endl;

    // Read configuration from command line

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "decoupled,d",
         po::bool_switch()->default_value(false),
         "Use decoupled aggregation"
        )
        (
         "size,n",
         po::value<ptrdiff_t>()->default_value(128),
         "domain size"
        )
        ("prm-file,P",
         po::value<std::string>(),
         "Parameter file in json format. "
        )
        (
         "prm,p",
         po::value< std::vector<std::string> >()->multitoken(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        if (comm.rank == 0) std::cout << desc << std::endl;
        return 0;
    }

    boost::property_tree::ptree prm;
    if (vm.count("prm-file")) {
        read_json(vm["prm-file"].as<std::string>(), prm);
    }

    if (vm.count("prm")) {
        BOOST_FOREACH(std::string v, vm["prm"].as<std::vector<std::string> >()) {
            amgcl::put(prm, v);
        }
    }

    ptrdiff_t n = vm["size"].as<ptrdiff_t>();

    boost::array<ptrdiff_t, 3> lo = { {0,   0,   0  } };
    boost::array<ptrdiff_t, 3> hi = { {n-1, n-1, n-1} };

    using amgcl::prof;

    prof.tic("partition");
    domain_partition<3> part(lo, hi, comm.size);
    ptrdiff_t chunk = part.size( comm.rank );

    std::vector<ptrdiff_t> domain = amgcl::mpi::exclusive_sum(comm, chunk);

    lo = part.domain(comm.rank).min_corner();
    hi = part.domain(comm.rank).max_corner();

    renumbering renum(part, domain);
    prof.toc("partition");

    prof.tic("assemble");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<val_type>  val;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 7);
    val.reserve(chunk * 7);

    ptr.push_back(0);

    const val_type h2i = (n - 1) * (n - 1) * math::identity<val_type>();

    for(ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
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

                ptr.push_back( col.size() );
            }
        }
    }
    prof.toc("assemble");

    if (vm["decoupled"].as<bool>()) {
        solve<amgcl::mpi::coarsening::smoothed_aggregation>(ptr, col, val, prm);
    } else {
        solve<amgcl::mpi::coarsening::smoothed_pmis>(ptr, col, val, prm);
    }
}
