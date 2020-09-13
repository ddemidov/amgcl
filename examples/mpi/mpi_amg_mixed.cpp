#include <iostream>
#include <vector>
#include <string>

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>

#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/preconditioner.hpp>
#include <amgcl/mpi/solver/runtime.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
    profiler<> prof;
}

namespace math = amgcl::math;

//---------------------------------------------------------------------------
ptrdiff_t assemble_poisson3d(amgcl::mpi::communicator comm,
        ptrdiff_t n, int block_size,
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs)
{
    ptrdiff_t n3 = n * n * n;

    ptrdiff_t chunk = (n3 + comm.size - 1) / comm.size;
    if (chunk % block_size != 0) {
        chunk += block_size - chunk % block_size;
    }
    ptrdiff_t row_beg = std::min(n3, chunk * comm.rank);
    ptrdiff_t row_end = std::min(n3, row_beg + chunk);
    chunk = row_end - row_beg;

    ptr.clear(); ptr.reserve(chunk + 1);
    col.clear(); col.reserve(chunk * 7);
    val.clear(); val.reserve(chunk * 7);

    rhs.resize(chunk);
    std::fill(rhs.begin(), rhs.end(), 1.0);

    const double h2i = (n - 1) * (n - 1);
    ptr.push_back(0);

    for (ptrdiff_t idx = row_beg; idx < row_end; ++idx) {
        ptrdiff_t k = idx / (n * n);
        ptrdiff_t j = (idx / n) % n;
        ptrdiff_t i = idx % n;

        if (k > 0)  {
            col.push_back(idx - n * n);
            val.push_back(-h2i);
        }

        if (j > 0)  {
            col.push_back(idx - n);
            val.push_back(-h2i);
        }

        if (i > 0) {
            col.push_back(idx - 1);
            val.push_back(-h2i);
        }

        col.push_back(idx);
        val.push_back(6 * h2i);

        if (i + 1 < n) {
            col.push_back(idx + 1);
            val.push_back(-h2i);
        }

        if (j + 1 < n) {
            col.push_back(idx + n);
            val.push_back(-h2i);
        }

        if (k + 1 < n) {
            col.push_back(idx + n * n);
            val.push_back(-h2i);
        }

        ptr.push_back( col.size() );
    }

    return chunk;
}

//---------------------------------------------------------------------------
template <class Backend, class Matrix>
std::shared_ptr< amgcl::mpi::distributed_matrix<Backend> >
partition(amgcl::mpi::communicator comm, const Matrix &Astrip,
        typename Backend::vector &rhs, const typename Backend::params &bprm,
        amgcl::runtime::mpi::partition::type ptype, int block_size = 1)
{
    typedef typename Backend::value_type val_type;
    typedef typename amgcl::math::rhs_of<val_type>::type rhs_type;
    typedef amgcl::mpi::distributed_matrix<Backend> DMatrix;

    using amgcl::prof;

    auto A = std::make_shared<DMatrix>(comm, Astrip);

    if (comm.size == 1 || ptype == amgcl::runtime::mpi::partition::merge)
        return A;

    prof.tic("partition");
    boost::property_tree::ptree prm;
    prm.put("type", ptype);
    prm.put("shrink_ratio", 1);
    amgcl::runtime::mpi::partition::wrapper<Backend> part(prm);

    auto I = part(*A, block_size);
    auto J = transpose(*I);
    A = product(*J, *product(*A, *I));

    amgcl::backend::numa_vector<rhs_type> new_rhs(J->loc_rows());

    J->move_to_backend(bprm);

    amgcl::backend::spmv(1, *J, rhs, 0, new_rhs);
    rhs.swap(new_rhs);
    prof.toc("partition");

    return A;
}

//---------------------------------------------------------------------------
void solve_scalar(
        amgcl::mpi::communicator comm,
        ptrdiff_t chunk,
        const std::vector<ptrdiff_t> &ptr,
        const std::vector<ptrdiff_t> &col,
        const std::vector<double> &val,
        const boost::property_tree::ptree &prm,
        const std::vector<double> &f,
        amgcl::runtime::mpi::partition::type ptype
        )
{
    typedef amgcl::backend::builtin<double> DBackend;
    typedef amgcl::backend::builtin<float>  SBackend;

    typedef
        amgcl::mpi::amg<
            SBackend,
            amgcl::runtime::mpi::coarsening::wrapper<SBackend>,
            amgcl::runtime::mpi::relaxation::wrapper<SBackend>,
            amgcl::runtime::mpi::direct::solver<float>,
            amgcl::runtime::mpi::partition::wrapper<SBackend>
            > Precond;
    typedef
        amgcl::runtime::mpi::solver::wrapper<DBackend>
        Solver;

    using amgcl::prof;

    typename DBackend::params bprm;

    amgcl::backend::numa_vector<double> rhs(f);

    prof.tic("setup");
    amgcl::mpi::distributed_matrix<DBackend> A(comm, std::tie(chunk, ptr, col, val));
    A.move_to_backend(bprm);
    Precond P(comm, std::tie(chunk, ptr, col, val), prm, bprm);
    Solver  S(chunk, prm.get_child("solver", amgcl::detail::empty_ptree()), bprm, amgcl::mpi::inner_product(comm));
    prof.toc("setup");

    if (comm.rank == 0) {
        std::cout << S << std::endl;
        std::cout << P << std::endl;
    }

    amgcl::backend::numa_vector<double> x(chunk);

    int    iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = S(A, P, rhs, x);
    prof.toc("solve");

    if (comm.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << error << std::endl
            << prof << std::endl;
    }
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator comm(MPI_COMM_WORLD);

    if (comm.rank == 0)
        std::cout << "World size: " << comm.size << std::endl;

    using amgcl::prof;

    // Read configuration from command line
    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "partitioner,r",
         po::value<amgcl::runtime::mpi::partition::type>()->default_value(
#if defined(AMGCL_HAVE_SCOTCH)
             amgcl::runtime::mpi::partition::ptscotch
#elif defined(AMGCL_HAVE_PASTIX)
             amgcl::runtime::mpi::partition::parmetis
#else
             amgcl::runtime::mpi::partition::merge
#endif
             ),
         "Repartition the system matrix"
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
        for(const std::string &v : vm["prm"].as<std::vector<std::string> >()) {
            amgcl::put(prm, v);
        }
    }

    ptrdiff_t n;
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    int aggr_block = prm.get("precond.coarsening.aggr.block_size", 1);

    amgcl::runtime::mpi::partition::type ptype = vm["partitioner"].as<amgcl::runtime::mpi::partition::type>();

    prof.tic("assemble");
    n = assemble_poisson3d(comm,
            vm["size"].as<ptrdiff_t>(),
            aggr_block, ptr, col, val, rhs);
    prof.toc("assemble");

    solve_scalar(comm, n, ptr, col, val, prm, rhs, ptype);
}
