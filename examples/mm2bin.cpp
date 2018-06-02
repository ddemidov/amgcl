#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <amgcl/util.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>

int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    namespace io = amgcl::io;

    using amgcl::precondition;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        ("input,i", po::value<std::string>()->required(),
             "Input matrix in the MatrixMarket format.")
        ("output,o", po::value<std::string>()->required(),
             "Output binary file.")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    io::mm_reader read(vm["input"].as<std::string>());
    precondition(!read.is_complex(), "Complex matrices are not supported!");
    precondition(!read.is_integer(), "Integer matrices are not supported!");

    std::ofstream f(vm["output"].as<std::string>().c_str(), std::ios::binary);
    precondition(f, "Failed to open output file for writing.");

    if (read.is_sparse()) {
        size_t rows, cols;
        std::vector<ptrdiff_t> ptr, col;
        std::vector<double>    val;

        std::tie(rows, cols) = read(ptr, col, val);

        precondition(io::write(f, rows), "File I/O error.");
        precondition(io::write(f, ptr),  "File I/O error.");
        precondition(io::write(f, col),  "File I/O error.");
        precondition(io::write(f, val),  "File I/O error.");

        std::cout
            << "Wrote " << rows << " by " << cols << " sparse matrix, "
            << ptr.back() << " nonzeros" << std::endl;
    } else {
        size_t rows, cols;
        std::vector<double> val;

        std::tie(rows, cols) = read(val);

        precondition(io::write(f, rows), "File I/O error.");
        precondition(io::write(f, cols), "File I/O error.");
        precondition(io::write(f, val),  "File I/O error.");

        std::cout
            << "Wrote " << rows << " by " << cols << " dense matrix"
            << std::endl;
    }
}
