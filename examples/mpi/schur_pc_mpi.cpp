#include <iostream>
#include <iterator>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/foreach.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/scope_exit.hpp>

#include <amgcl/io/binary.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/runtime.hpp>
#include <amgcl/mpi/make_solver.hpp>
#include <amgcl/mpi/schur_pressure_correction.hpp>
#include <amgcl/mpi/block_preconditioner.hpp>
#include <amgcl/mpi/subdomain_deflation.hpp>
#include <amgcl/mpi/direct_solver.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
    profiler<> prof;
}

using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
std::vector<ptrdiff_t> read_problem(
        const amgcl::mpi::communicator &world,
        const std::string &A_file,
        const std::string &rhs_file,
        const std::string &part_file,
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs
        )
{
    // Read partition
    ptrdiff_t n, m;
    std::vector<ptrdiff_t> domain(world.size + 1, 0);
    std::vector<int> part;

    amgcl::io::read_dense(part_file, n, m, part);
    BOOST_FOREACH(int p, part) {
        ++domain[p+1];
        precondition(p < world.size, "MPI world does not correspond to partition");
    }
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    ptrdiff_t chunk_beg = domain[world.rank];
    ptrdiff_t chunk_end = domain[world.rank + 1];
    ptrdiff_t chunk     = chunk_end - chunk_beg;

    // Reorder unknowns
    std::vector<ptrdiff_t> order(n);
    for(ptrdiff_t i = 0; i < n; ++i)
        order[i] = domain[part[i]]++;

    std::rotate(domain.begin(), domain.end(), domain.end() - 1);
    domain[0] = 0;

    // Read matrix chunk
    {
        using namespace amgcl::io;

        std::ifstream A(A_file.c_str(), std::ios::binary);
        precondition(A, "Failed to open matrix file (" + A_file + ")");

        std::ifstream b(rhs_file.c_str(), std::ios::binary);
        precondition(b, "Failed to open rhs file (" + rhs_file + ")");

        ptrdiff_t rows;
        precondition(read(A, rows), "File I/O error");
        precondition(rows == n, "Matrix and partition have incompatible sizes");

        ptr.clear(); ptr.reserve(chunk + 1); ptr.push_back(0);

        std::vector<ptrdiff_t> gptr(n + 1);
        precondition(read(A, gptr), "File I/O error");

        size_t col_beg = sizeof(rows) + sizeof(gptr[0]) * (n + 1);
        size_t val_beg = col_beg + sizeof(col[0]) * gptr.back();
        size_t rhs_beg = 2 * sizeof(ptrdiff_t);

        // Count local nonzeros
        for(ptrdiff_t i = 0; i < n; ++i)
            if (part[i] == world.rank)
                ptr.push_back(gptr[i+1] - gptr[i]);

        std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

        col.clear(); col.reserve(ptr.back());
        val.clear(); val.reserve(ptr.back());
        rhs.clear(); rhs.reserve(chunk);

        // Read local matrix and rhs stripes
        for(ptrdiff_t i = 0; i < n; ++i) {
            if (part[i] != world.rank) continue;

            ptrdiff_t c;
            A.seekg(col_beg + gptr[i] * sizeof(c));
            for(ptrdiff_t j = gptr[i], e = gptr[i+1]; j < e; ++j) {
                precondition(read(A, c), "File I/O error (1)");
                col.push_back(order[c]);
            }
        }

        for(ptrdiff_t i = 0; i < n; ++i) {
            if (part[i] != world.rank) continue;

            double v;
            A.seekg(val_beg + gptr[i] * sizeof(v));
            for(ptrdiff_t j = gptr[i], e = gptr[i+1]; j < e; ++j) {
                precondition(read(A, v), "File I/O error (2)");
                val.push_back(v);
            }
        }

        for(ptrdiff_t i = 0; i < n; ++i) {
            if (part[i] != world.rank) continue;

            double f;
            b.seekg(rhs_beg + i * sizeof(f));
            precondition(read(b, f), "File I/O error (3)");
            rhs.push_back(f);
        }
    }

    return domain;
}

//---------------------------------------------------------------------------
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
    namespace po = boost::program_options;
    using std::string;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "binary,B",
         po::bool_switch()->default_value(false),
         "When specified, treat input files as binary instead of as MatrixMarket. "
         "It is assumed the files were converted to binary format with mm2bin utility. "
        )
        (
         "matrix,A",
         po::value<string>()->required(),
         "The system matrix in MatrixMarket format"
        )
        (
         "rhs,f",
         po::value<string>(),
         "The right-hand side in MatrixMarket format"
        )
        (
         "part,s",
         po::value<string>()->required(),
         "Partitioning of the problem in MatrixMarket format"
        )
        (
         "pmask,m",
         po::value<string>()->required(),
         "The pressure mask in MatrixMarket format. Or, if the parameter has "
         "the form '%n:m', then each (n+i*m)-th variable is treated as pressure."
        )
        (
         "params,P",
         po::value<string>(),
         "parameter file in json format"
        )
        (
         "prm,p",
         po::value< std::vector<string> >()->multitoken(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        if (world.rank == 0)
            std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    boost::property_tree::ptree prm;
    if (vm.count("params")) read_json(vm["params"].as<string>(), prm);

    if (vm.count("prm")) {
        BOOST_FOREACH(string pair, vm["prm"].as<std::vector<string> >())
        {
            using namespace boost::algorithm;
            std::vector<string> key_val;
            split(key_val, pair, is_any_of("="));
            if (key_val.size() != 2) throw po::invalid_option_value(
                    "Parameters specified with -p option "
                    "should have name=value format");

            prm.put(key_val[0], key_val[1]);
        }
    }

    prof.tic("read problem");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    std::vector<ptrdiff_t> domain = read_problem(
            world,
            vm["matrix"].as<string>(), vm["rhs"].as<string>(), vm["part"].as<string>(),
            ptr, col, val, rhs
            );

    ptrdiff_t chunk = domain[world.rank + 1] - domain[world.rank];
    prof.toc("read problem");

    std::vector<char> pm(chunk, 0);
    for(ptrdiff_t i = 0; i < chunk; i += 4) pm[i] = 1;
    prm.put("precond.pmask", static_cast<void*>(&pm[0]));
    prm.put("precond.pmask_size", chunk);

    prof.tic("setup");
    typedef amgcl::backend::builtin<double> Backend;

    typedef
        amgcl::mpi::make_solver<
            amgcl::mpi::schur_pressure_correction<
                amgcl::mpi::make_solver<
                    amgcl::mpi::block_preconditioner<
                        amgcl::runtime::relaxation::as_preconditioner<Backend>
                        >,
                    amgcl::runtime::iterative_solver
                    >,
                amgcl::mpi::make_solver<
                    amgcl::mpi::subdomain_deflation<
                        amgcl::runtime::amg<Backend>,
                        amgcl::runtime::iterative_solver,
                        amgcl::runtime::mpi::direct_solver<double>
                        >,
                    amgcl::runtime::iterative_solver
                    >
                >,
            amgcl::runtime::iterative_solver
            > Solver;

    boost::function<double(ptrdiff_t,unsigned)> dv = amgcl::mpi::constant_deflation(1);
    prm.put("precond.psolver.precond.num_def_vec", 1);
    prm.put("precond.psolver.precond.def_vec", &dv);

    Solver solve(world, boost::tie(chunk, ptr, col, val), prm);
    prof.toc("setup");

    std::vector<double> x(chunk, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(rhs, x);
    prof.toc("solve");

    if (world.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << resid << std::endl
            << std::endl
            << prof << std::endl;
    }
}
