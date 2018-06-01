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
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/solver/runtime.hpp>

#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/amg.hpp>
#include <amgcl/mpi/coarsening/runtime.hpp>
#include <amgcl/mpi/relaxation/runtime.hpp>
#include <amgcl/mpi/direct_solver/runtime.hpp>
#include <amgcl/mpi/repartition/runtime.hpp>

#include <amgcl/io/mm.hpp>
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
template <class Backend, class Matrix, typename rhs_type>
boost::shared_ptr< amgcl::mpi::distributed_matrix<Backend> >
partition(amgcl::mpi::communicator comm, const Matrix &Astrip,
        std::vector<rhs_type> &rhs,
        amgcl::runtime::mpi::repartition::type ptype, int block_size = 1)
{
    typedef amgcl::mpi::distributed_matrix<Backend> DMatrix;

    using amgcl::prof;

    boost::shared_ptr<DMatrix> A = boost::make_shared<DMatrix>(comm, Astrip);

    if (comm.size == 1 || ptype == amgcl::runtime::mpi::repartition::dummy)
        return A;

    prof.tic("partition");
    boost::property_tree::ptree prm;
    prm.put("type", ptype);
    prm.put("shrink_ratio", 1);
    amgcl::runtime::mpi::repartition::wrapper<Backend> part(prm);

    boost::shared_ptr<DMatrix> I = part(*A, block_size);
    boost::shared_ptr<DMatrix> J = transpose(*I);
    A = product(*J, *product(*A, *I));

    std::vector<rhs_type> new_rhs(J->loc_rows());

    J->move_to_backend();

    amgcl::backend::spmv(1, *J, rhs, 0, new_rhs);
    rhs.swap(new_rhs);
    prof.toc("partition");

    return A;
}

//---------------------------------------------------------------------------
template <int B>
void solve_block(
        amgcl::mpi::communicator comm,
        ptrdiff_t chunk,
        const std::vector<ptrdiff_t>      &ptr,
        const std::vector<ptrdiff_t>      &col,
        const std::vector<double>         &val,
        const boost::property_tree::ptree &prm,
        const std::vector<double>         &f,
        amgcl::runtime::mpi::repartition::type ptype
        )
{
    typedef amgcl::static_matrix<double, B, B> val_type;
    typedef amgcl::static_matrix<double, B, 1> rhs_type;

    typedef amgcl::backend::builtin<val_type> Backend;
    typedef amgcl::mpi::distributed_matrix<Backend> Matrix;

    typedef
        amgcl::mpi::make_solver<
            amgcl::mpi::amg<
                Backend,
                amgcl::runtime::mpi::coarsening::wrapper<Backend>,
                amgcl::runtime::mpi::relaxation::wrapper<Backend>,
                amgcl::runtime::mpi::direct::solver<val_type>,
                amgcl::runtime::mpi::repartition::wrapper<Backend>
                >,
            amgcl::runtime::solver::wrapper
            >
        Solver;

    using amgcl::prof;

    std::vector<rhs_type> rhs(
            reinterpret_cast<const rhs_type*>(&f[0]),
            reinterpret_cast<const rhs_type*>(&f[0]) + chunk / B
            );

    boost::shared_ptr<Matrix> A = partition<Backend>(comm,
            amgcl::adapter::block_matrix<val_type>(boost::tie(chunk, ptr, col, val)),
            rhs, ptype, prm.get("precond.coarsening.aggr.block_size", 1));

    prof.tic("setup");
    Solver solve(comm, A, prm);
    prof.toc("setup");

    if (comm.rank == 0) {
        std::cout << solve << std::endl;
    }

    std::vector<rhs_type> x(A->loc_rows(), math::zero<rhs_type>());

    int    iters;
    double error;

    prof.tic("solve");
    boost::tie(iters, error) = solve(rhs, x);
    prof.toc("solve");

    if (comm.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << error << std::endl
            << prof << std::endl;
    }
}
//---------------------------------------------------------------------------
void solve_scalar(
        amgcl::mpi::communicator comm,
        ptrdiff_t chunk,
        const std::vector<ptrdiff_t> &ptr,
        const std::vector<ptrdiff_t> &col,
        const std::vector<double> &val,
        const boost::property_tree::ptree &prm,
        std::vector<double> &rhs,
        amgcl::runtime::mpi::repartition::type ptype
        )
{
    typedef amgcl::backend::builtin<double> Backend;
    typedef amgcl::mpi::distributed_matrix<Backend> Matrix;

    typedef
        amgcl::mpi::make_solver<
            amgcl::mpi::amg<
                Backend,
                amgcl::runtime::mpi::coarsening::wrapper<Backend>,
                amgcl::runtime::mpi::relaxation::wrapper<Backend>,
                amgcl::runtime::mpi::direct::solver<double>,
                amgcl::runtime::mpi::repartition::wrapper<Backend>
                >,
            amgcl::runtime::solver::wrapper
            >
        Solver;

    using amgcl::prof;

    boost::shared_ptr<Matrix> A = partition<Backend>(comm,
            boost::tie(chunk, ptr, col, val), rhs, ptype,
            prm.get("precond.coarsening.aggr.block_size", 1));

    prof.tic("setup");
    Solver solve(comm, A, prm);
    prof.toc("setup");

    if (comm.rank == 0) {
        std::cout << solve << std::endl;
    }

    std::vector<double> x(A->loc_rows(), amgcl::math::zero<double>());

    int    iters;
    double error;

    prof.tic("solve");
    boost::tie(iters, error) = solve(rhs, x);
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
         "block-size,b",
         po::value<int>()->default_value(1),
         "The block size of the system matrix. "
         "When specified, the system matrix is assumed to have block-wise structure. "
         "This usually is the case for problems in elasticity, structural mechanics, "
         "for coupled systems of PDE (such as Navier-Stokes equations), etc. "
        )
        (
         "partitioner,r",
         po::value<amgcl::runtime::mpi::repartition::type>()->default_value(
#if defined(AMGCL_HAVE_SCOTCH)
             amgcl::runtime::mpi::repartition::scotch
#elif defined(AMGCL_HAVE_PASTIX)
             amgcl::runtime::mpi::repartition::parmetis
#else
             amgcl::runtime::mpi::repartition::dummy
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
        BOOST_FOREACH(std::string v, vm["prm"].as<std::vector<std::string> >()) {
            amgcl::put(prm, v);
        }
    }

    ptrdiff_t n;
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    int block_size = vm["block-size"].as<int>();
    int aggr_block = prm.get("precond.coarsening.aggr.block_size", 1);

    if (vm.count("matrix")) {
        prof.tic("read");
        n = read_matrix_market(comm,
                vm["matrix"].as<std::string>(),
                vm["rhs"].as<std::string>(),
                block_size * aggr_block, ptr, col, val, rhs);
        prof.toc("read");
    } else {
        prof.tic("assemble");
        n = assemble_poisson3d(comm,
                vm["size"].as<ptrdiff_t>(),
                block_size * aggr_block, ptr, col, val, rhs);
        prof.toc("assemble");
    }

    amgcl::runtime::mpi::repartition::type ptype = vm["partitioner"].as<amgcl::runtime::mpi::repartition::type>();

    switch(block_size) {
#define AMGCL_CALL_BLOCK_SOLVER(z, data, B)                                \
        case B:                                                            \
            solve_block<B>(comm, n, ptr, col, val, prm, rhs, ptype);       \
            break;

        BOOST_PP_SEQ_FOR_EACH(AMGCL_CALL_BLOCK_SOLVER, ~, AMGCL_BLOCK_SIZES)

#undef AMGCL_CALL_BLOCK_SOLVER

        case 1:
            solve_scalar(comm, n, ptr, col, val, prm, rhs, ptype);
            break;
        default:
            if (comm.rank == 0)
                std::cout << "Unsupported block size!" << std::endl;
    }
}
