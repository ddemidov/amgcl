#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/scope_exit.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/block_preconditioner.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

typedef amgcl::scoped_tic< amgcl::profiler<> > scoped_tic;

//---------------------------------------------------------------------------
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

//---------------------------------------------------------------------------
template <template <class> class Precond, class Matrix>
boost::tuple<size_t, double> solve(
        const amgcl::mpi::communicator &comm,
        const boost::property_tree::ptree &prm,
        const Matrix &A
        )
{
    typedef amgcl::backend::builtin<double> Backend;

    typedef amgcl::mpi::make_solver<
        amgcl::mpi::block_preconditioner< Precond<Backend> >,
        amgcl::runtime::iterative_solver
        > Solver;

    const size_t n = amgcl::backend::rows(A);

    std::vector<double> rhs(n, 1), x(n, 0);

    scoped_tic t1(prof, "setup");
    Solver solve(comm, A, prm);

    {
        scoped_tic t2(prof, "solve");
        return solve(rhs, x);
    }
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;

    using amgcl::prof;
    using std::vector;
    using std::string;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        ("prm-file,P",
         po::value<string>(),
         "Parameter file in json format. "
        )
        (
         "prm,p",
         po::value< vector<string> >()->multitoken(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        (
         "size,n",
         po::value<int>()->default_value(1024),
         "The size of the Poisson problem to solve. "
         "Specified as number of grid nodes along each dimension of a unit square. "
         "The resulting system will have n*n unknowns. "
        )
        (
         "single-level,1",
         po::bool_switch()->default_value(false),
         "When specified, the AMG hierarchy is not constructed. "
         "Instead, the problem is solved using a single-level smoother as preconditioner. "
        )
        (
         "initial,x",
         po::value<double>()->default_value(0),
         "Value to use as initial approximation. "
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
    if (vm.count("prm-file")) {
        read_json(vm["prm-file"].as<string>(), prm);
    }

    if (vm.count("prm")) {
        BOOST_FOREACH(string pair, vm["prm"].as<vector<string> >())
        {
            using namespace boost::algorithm;
            vector<string> key_val;
            split(key_val, pair, is_any_of("="));
            if (key_val.size() != 2) throw po::invalid_option_value(
                    "Parameters specified with -p option "
                    "should have name=value format");

            prm.put(key_val[0], key_val[1]);
        }
    }

    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    const ptrdiff_t n   = vm["size"].as<int>();
    const double    h2i = (n - 1) * (n - 1);

    boost::array<ptrdiff_t, 2> lo = { {0, 0} };
    boost::array<ptrdiff_t, 2> hi = { {n - 1, n - 1} };

    prof.tic("partition");
    domain_partition<2> part(lo, hi, world.size);
    ptrdiff_t chunk = part.size( world.rank );

    std::vector<ptrdiff_t> domain(world.size + 1);
    MPI_Allgather(
            &chunk, 1, amgcl::mpi::datatype<ptrdiff_t>(),
            &domain[1], 1, amgcl::mpi::datatype<ptrdiff_t>(), world);
    boost::partial_sum(domain, domain.begin());

    lo = part.domain(world.rank).min_corner();
    hi = part.domain(world.rank).max_corner();
    prof.toc("partition");

    renumbering renum(part, domain);

    prof.tic("assemble");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 5);
    val.reserve(chunk * 5);

    ptr.push_back(0);

    for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
        for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
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

            ptr.push_back( col.size() );
        }
    }
    prof.toc("assemble");

    size_t iters;
    double error;

    bool single_level = vm["single-level"].as<bool>();

    if (single_level) {
        boost::tie(iters, error) = solve<amgcl::runtime::relaxation::as_preconditioner>(
                world, prm, boost::tie(chunk, ptr, col, val));
    } else {
        boost::tie(iters, error) = solve<amgcl::runtime::amg>(
                world, prm, boost::tie(chunk, ptr, col, val));
    }

    if (world.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << error << std::endl
            << std::endl
            << prof << std::endl;
    }
}
