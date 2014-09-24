#include <iostream>
#include <iterator>
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

namespace amgcl {
    profiler<> prof;
}

//---------------------------------------------------------------------------
inline size_t alignup(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

//---------------------------------------------------------------------------
std::vector<ptrdiff_t> read_problem(
        const amgcl::mpi::communicator &world,
        const std::string &A_file,
        const std::string &rhs_file,
        int block_size,
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs
        )
{
    using amgcl::precondition;

    std::ifstream A(A_file.c_str());
    precondition(A, "Failed to open matrix file (" + A_file + ")");

    std::string line;
    ptrdiff_t n = 0, nnz = 0;
    while ( std::getline(A, line) ) {
        if (line[0] == '%') continue;
        std::istringstream is(line);
        ptrdiff_t m;
        precondition(is >> n >> m >> nnz, "Unsupported format in matrix file");
        precondition(n == m, "Non-square matrix in matrix file");
        break;
    }

    precondition(n, "Empty matrix file");
    precondition(n % block_size == 0, "Matrix size is not divisible by block_size");

    ptrdiff_t chunk = alignup((n + world.size - 1) / world.size, block_size);
    std::vector<ptrdiff_t> domain(world.size + 1);
    domain[0] = 0;
    for(int i = 0; i < world.size; ++i)
        domain[i+1] = std::min(domain[i] + chunk, n);

    ptrdiff_t chunk_beg = domain[world.rank];
    ptrdiff_t chunk_end = domain[world.rank + 1];
    chunk = chunk_end - chunk_beg;

    {
        std::vector<ptrdiff_t> I, J;
        std::vector<double>    V;

        ptr.clear();
        ptr.resize(chunk + 1, 0);

        while (std::getline(A, line)) {
            if (line[0] == '%') continue;
            std::istringstream is(line);
            ptrdiff_t i, j;
            double v;
            precondition(is >> i >> j >> v, "Unsupported format in matrix file");
            --i;
            --j;

            if (i < chunk_beg || i >= chunk_end) continue;

            ++ptr[i + 1 - chunk_beg];

            I.push_back(i);
            J.push_back(j);
            V.push_back(v);
        }

        boost::partial_sum(ptr, ptr.begin());

        ptrdiff_t loc_nnz = ptr.back();

        col.clear(); col.resize(loc_nnz);
        val.clear(); val.resize(loc_nnz);

        for(ptrdiff_t i = 0; i < loc_nnz; ++i) {
            ptrdiff_t row = I[i] - chunk_beg;
            col[ptr[row]] = J[i];
            val[ptr[row]] = V[i];
            ++ptr[row];
        }
        std::rotate(ptr.begin(), ptr.end() - 1, ptr.end());
        ptr[0] = 0;
    }

    std::ifstream f(rhs_file.c_str());
    precondition(f, "Failed to open rhs file (" + rhs_file + ")");

    while (std::getline(f, line)) {
        if (line[0] == '%') continue;
        std::istringstream is(line);
        ptrdiff_t rows, cols;
        precondition(is >> rows >> cols, "Unsupported format in matrix file");
        precondition(rows == n, "RHS size should coincide with matrix size");
        break;
    }

    rhs.clear();
    rhs.reserve(chunk);

    ptrdiff_t pos = 0;
    while (std::getline(f, line)) {
        if (line[0] == '%') continue;
        std::istringstream is(line);
        double v;
        precondition(is >> v, "Unsupported format in RHS file");

        if (pos++ < chunk_beg) continue;
        if (pos >= chunk_end) break;

        rhs.push_back(v);
    }

    return domain;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    // Read configuration from command line
    amgcl::runtime::coarsening::type    coarsening       = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type    relaxation       = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type        iterative_solver = amgcl::runtime::solver::bicgstabl;
    amgcl::runtime::direct_solver::type direct_solver    = amgcl::runtime::direct_solver::skyline_lu;
    std::string parameter_file;
    std::string A_file   = "A.mm";
    std::string rhs_file = "rhs.mm";
    std::string out_file;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "coarsening,c",
         po::value<amgcl::runtime::coarsening::type>(&coarsening)->default_value(coarsening),
         "ruge_stuben, aggregation, smoothed_aggregation, smoothed_aggr_emin"
        )
        (
         "relaxation,r",
         po::value<amgcl::runtime::relaxation::type>(&relaxation)->default_value(relaxation),
         "gauss_seidel, ilu0, damped_jacobi, spai0, chebyshev"
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
         "params,p",
         po::value<std::string>(&parameter_file),
         "parameter file in json format"
        )
        (
         "matrix,A",
         po::value<std::string>(&A_file)->default_value(A_file),
         "The system matrix in MatrixMarket format"
        )
        (
         "rhs,b",
         po::value<std::string>(&rhs_file)->default_value(rhs_file),
         "The right-hand side in MatrixMarket format"
        )
        (
         "output,o",
         po::value<std::string>(&out_file),
         "The output file (saved in MatrixMarket format)"
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

    using amgcl::prof;

    int block_size = prm.get("amg.coarsening.aggr.block_size", 1);

    prof.tic("read problem");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    std::vector<ptrdiff_t> domain = read_problem(
            world, A_file, rhs_file, block_size, ptr, col, val, rhs
            );

    ptrdiff_t chunk = domain[world.rank + 1] - domain[world.rank];
    prof.toc("read problem");

    prof.tic("setup");
    typedef
        amgcl::runtime::mpi::subdomain_deflation<
            amgcl::backend::builtin<double>
            >
        SDD;

    SDD solve(
            coarsening, relaxation, iterative_solver, direct_solver,
            world, boost::tie(chunk, ptr, col, val),
            amgcl::mpi::constant_deflation(block_size), prm
            );
    double tm_setup = prof.toc("setup");

    std::vector<double> x(chunk, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(rhs, x);
    double tm_solve = prof.toc("solve");

    if (vm.count("output")) {
        prof.tic("save");
        for(int r = 0; r < world.size; ++r) {
            if (r == world.rank) {
                std::ofstream f(out_file.c_str(), std::ios::app);

                if (world.rank == 0)
                    f << domain.back() << " 1\n";

                std::ostream_iterator<double> oi(f, "\n");
                boost::copy(x, oi);
            }
            MPI_Barrier(world);
        }
        prof.toc("save");
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
        log_name << "log_" << domain.back() << "_" << nt << "_" << world.size << ".txt";
        std::ofstream log(log_name.str().c_str(), std::ios::app);
        log << domain.back() << "\t" << nt << "\t" << world.size
            << "\t" << tm_setup << "\t" << tm_solve
            << "\t" << iters << "\t" << std::endl;
    }

}
