#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/scope_exit.hpp>

#include <amgcl/io/binary.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/cpr.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/coarsening/runtime.hpp>
#include <amgcl/mpi/relaxation/runtime.hpp>
#include <amgcl/mpi/solver/runtime.hpp>
#include <amgcl/mpi/relaxation/as_preconditioner.hpp>
#include <amgcl/mpi/direct_solver/runtime.hpp>
#include <amgcl/mpi/partition/runtime.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
    profiler<> prof;
}

using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
ptrdiff_t read_matrix_market(
        amgcl::mpi::communicator comm,
        const std::string &A_file, const std::string &rhs_file, int block_size,
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs)
{
    amgcl::io::mm_reader A_mm(A_file);
    ptrdiff_t n = A_mm.rows();

    ptrdiff_t chunk = (n + comm.size - 1) / comm.size;
    if (chunk % block_size != 0) {
        chunk += block_size - chunk % block_size;
    }

    ptrdiff_t row_beg = std::min(n, chunk * comm.rank);
    ptrdiff_t row_end = std::min(n, row_beg + chunk);

    chunk = row_end - row_beg;

    A_mm(ptr, col, val, row_beg, row_end);

    if (rhs_file.empty()) {
        rhs.resize(chunk);
        std::fill(rhs.begin(), rhs.end(), 1.0);
    } else {
        amgcl::io::mm_reader rhs_mm(rhs_file);
        rhs_mm(rhs, row_beg, row_end);
    }

    return chunk;
}

//---------------------------------------------------------------------------
ptrdiff_t read_binary(
        amgcl::mpi::communicator comm,
        const std::string &A_file, const std::string &rhs_file, int block_size,
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs)
{
    ptrdiff_t n = amgcl::io::crs_size<ptrdiff_t>(A_file);

    ptrdiff_t chunk = (n + comm.size - 1) / comm.size;
    if (chunk % block_size != 0) {
        chunk += block_size - chunk % block_size;
    }

    ptrdiff_t row_beg = std::min(n, chunk * comm.rank);
    ptrdiff_t row_end = std::min(n, row_beg + chunk);

    chunk = row_end - row_beg;

    amgcl::io::read_crs(A_file, n, ptr, col, val, row_beg, row_end);

    if (rhs_file.empty()) {
        rhs.resize(chunk);
        std::fill(rhs.begin(), rhs.end(), 1.0);
    } else {
        ptrdiff_t rows, cols;
        amgcl::io::read_dense(rhs_file, rows, cols, rhs, row_beg, row_end);
    }

    return chunk;
}

//---------------------------------------------------------------------------
template <class Backend, class Matrix>
std::shared_ptr< amgcl::mpi::distributed_matrix<Backend> >
partition(amgcl::mpi::communicator comm, const Matrix &Astrip,
        std::vector<double> &rhs, const typename Backend::params &bprm,
        amgcl::runtime::mpi::partition::type ptype, int block_size = 1)
{
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

    std::vector<double> new_rhs(J->loc_rows());

    J->move_to_backend(bprm);

    amgcl::backend::spmv(1, *J, rhs, 0, new_rhs);
    rhs.swap(new_rhs);
    prof.toc("partition");

    return A;
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
        ("matrix,A",
         po::value<std::string>(),
         "System matrix in the MatrixMarket format. "
         "When not specified, a Poisson problem in 3D unit cube is assembled. "
        )
        (
         "rhs,f",
         po::value<std::string>()->default_value(""),
         "The RHS vector in the MatrixMarket format. "
         "When omitted, a vector of ones is used by default. "
         "Should only be provided together with a system matrix. "
        )
        (
         "binary,B",
         po::bool_switch()->default_value(false),
         "When specified, treat input files as binary instead of as MatrixMarket. "
         "It is assumed the files were converted to binary format with mm2bin utility. "
        )
        (
         "block-size,b",
         po::value<int>()->default_value(1),
         "The block size of the system matrix. "
        )
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

    int block_size = vm["block-size"].as<int>();
    prm.put("precond.block_size", block_size);

    prof.tic("read");
    if (vm["binary"].as<bool>()) {
        n = read_binary(comm,
                vm["matrix"].as<std::string>(),
                vm["rhs"].as<std::string>(),
                block_size, ptr, col, val, rhs);
    } else {
        n = read_matrix_market(comm,
                vm["matrix"].as<std::string>(),
                vm["rhs"].as<std::string>(),
                block_size, ptr, col, val, rhs);
    }
    prof.toc("read");

    typedef amgcl::backend::builtin<double> Backend;

    auto A = partition<Backend>(comm,
            std::tie(n, ptr, col, val), rhs, Backend::params(),
            vm["partitioner"].as<amgcl::runtime::mpi::partition::type>(),
            block_size);

    prof.tic("setup");
    typedef
        amgcl::mpi::make_solver<
            amgcl::mpi::cpr<
                amgcl::mpi::amg<
                    Backend,
                    amgcl::runtime::mpi::coarsening::wrapper<Backend>,
                    amgcl::runtime::mpi::relaxation::wrapper<Backend>,
                    amgcl::runtime::mpi::direct::solver<double>,
                    amgcl::runtime::mpi::partition::wrapper<Backend>
                    >,
                amgcl::mpi::relaxation::as_preconditioner<
                    amgcl::runtime::mpi::relaxation::wrapper<Backend>
                    >
                >,
            amgcl::runtime::mpi::solver::wrapper<Backend>
            >
        Solver;

    Solver solve(comm, A, prm);
    prof.toc("setup");

    if (comm.rank == 0)
        std::cout << solve << std::endl;

    std::vector<double> x(rhs.size(), 0.0);

    prof.tic("solve");
    size_t iters;
    double error;
    std::tie(iters, error) = solve(rhs, x);
    prof.toc("solve");

    if (comm.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << error << std::endl
            << prof << std::endl;
    }
}
