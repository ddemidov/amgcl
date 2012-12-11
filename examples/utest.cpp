#include <iostream>
#include <cstdlib>

#include <boost/program_options.hpp>

#include <vexcl/vexcl.hpp>

#define AMGCL_PROFILING

#include <amgcl/amgcl.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/interp_aggr.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/interp_classic.hpp>
#include <amgcl/level_vexcl.hpp>
#include <amgcl/operations_vexcl.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>

#include "read.hpp"

namespace po = boost::program_options;

namespace amgcl {
    profiler<> prof;
}
using amgcl::prof;

enum interp_t {
    classic              = 1,
    aggregation          = 2,
    smoothed_aggregation = 3
};

enum solver_t {
    cg   = 1,
    bicg = 2
};

struct options {
    int         interp;
    int         solver;
    std::string pfile;
};

//---------------------------------------------------------------------------
template <class AMG, class spmat, class vector>
void solve(
        const vex::Context &ctx,
        const AMG          &amg,
        const spmat        &A,
        const vector       &rhs,
        const options      &op
        )
{
    const int n = amgcl::sparse::matrix_rows(A);

    // Copy matrix and rhs to GPU(s).
    vex::SpMat<double, int, int> Agpu(
            ctx.queue(), n, n,
            amgcl::sparse::matrix_outer_index(A),
            amgcl::sparse::matrix_inner_index(A),
            amgcl::sparse::matrix_values(A)
            );

    vex::vector<double> f(ctx.queue(), rhs);
    vex::vector<double> x(ctx.queue(), n);
    x = 0;

    std::pair<int,double> cnv;
    prof.tic("Solve");
    switch (static_cast<solver_t>(op.solver)) {
        case cg:
            cnv = amgcl::solve(Agpu, f, amg, x, amgcl::cg_tag());
            break;
        case bicg:
            cnv = amgcl::solve(Agpu, f, amg, x, amgcl::bicg_tag());
            break;
        default:
            throw std::invalid_argument("Unsupported iterative solver");
    }
    prof.toc("Solve");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;
}

//---------------------------------------------------------------------------
template <class spmat, class vector>
void test_classic(const vex::Context &ctx,
        const spmat &A, const vector &rhs,
        const options &op
        )
{
    typedef amgcl::solver<
        double, int,
        amgcl::interp::classic,
        amgcl::level::vexcl
    > AMG;

    typename AMG::params prm;


    prof.tic("Setup");
    AMG amg(A, prm);
    prof.toc("Setup");

    solve(ctx, amg, A, rhs, op);
}

//---------------------------------------------------------------------------
template <class spmat, class vector>
void test_aggregation(const vex::Context &ctx,
        const spmat &A, const vector &rhs,
        const options &op
        )
{
    typedef amgcl::solver<
        double, int,
        amgcl::interp::aggregation<amgcl::aggr::plain>,
        amgcl::level::vexcl
    > AMG;

    typename AMG::params prm;

    const int n = amgcl::sparse::matrix_rows(A);

    prof.tic("Setup");
    AMG amg(A, prm);
    prof.toc("Setup");

    solve(ctx, amg, A, rhs, op);
}

//---------------------------------------------------------------------------
template <class spmat, class vector>
void test_smoothed_aggregation(const vex::Context &ctx,
        const spmat &A, const vector &rhs,
        const options &op
        )
{
    typedef amgcl::solver<
        double, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::vexcl
    > AMG;

    typename AMG::params prm;

    const int n = amgcl::sparse::matrix_rows(A);

    prof.tic("Setup");
    AMG amg(A, prm);
    prof.toc("Setup");

    solve(ctx, amg, A, rhs, op);
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    options op;

    po::options_description desc("Possible options");

    desc.add_options()
        ("help", "Show help")
        ("interp",
            po::value<int>(&op.interp)->default_value(smoothed_aggregation),
            "Interpolation: classic(1), aggregation(2), smoothed_aggregation (3)"
            )
        ("solver", po::value<int>(&op.solver)->default_value(cg),
            "Iterative solver: cg(1), bicgstab(2)")
        ("problem",
            po::value<std::string>(&op.pfile)->default_value("problem.dat"),
            "Problem file"
            )
        ;

    po::positional_options_description pdesc;
    pdesc.add("problem", -1);


    po::variables_map vm;
    po::store(
            po::command_line_parser(argc, argv).
                options(desc).
                positional(pdesc).
                run(),
            vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    prof.tic("OpenCL initialization");
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    prof.toc("OpenCL initialization");

    if (!ctx.size()) throw std::runtime_error("No available compute devices");
    std::cout << ctx << std::endl;

    prof.tic("Read problem");
    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> rhs;
    int n = read_problem(op.pfile, row, col, val, rhs);
    prof.toc("Read problem");

    auto A = amgcl::sparse::map(n, n, row.data(), col.data(), val.data());

    switch(static_cast<interp_t>(op.interp)) {
        case classic:
            test_classic(ctx, A, rhs, op);
            break;
        case aggregation:
            test_aggregation(ctx, A, rhs, op);
            break;
        case smoothed_aggregation:
            test_smoothed_aggregation(ctx, A, rhs, op);
            break;
        default:
            throw std::invalid_argument("Unsupported interpolation scheme");
    }

    std::cout << prof << std::endl;
}
