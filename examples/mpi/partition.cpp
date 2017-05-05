#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <boost/program_options.hpp>

#include <amgcl/util.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>

extern "C" {
#include <metis.h>
}

using amgcl::precondition;

//---------------------------------------------------------------------------
void pointwise_graph(
        int n, int block_size,
        const std::vector<int> &ptr,
        const std::vector<int> &col,
        std::vector<int> &pptr,
        std::vector<int> &pcol
        )
{
    int np = n / block_size;

    assert(np * block_size == n);

    // Create pointwise matrix
    std::vector<int> ptr1(np + 1, 0);
    std::vector<int> marker(np, -1);
    for(int ip = 0, i = 0; ip < np; ++ip) {
        for(int k = 0; k < block_size; ++k, ++i) {
            for(int j = ptr[i]; j < ptr[i+1]; ++j) {
                int cp = col[j] / block_size;
                if (marker[cp] != ip) {
                    marker[cp] = ip;
                    ++ptr1[ip+1];
                }
            }
        }
    }

    std::partial_sum(ptr1.begin(), ptr1.end(), ptr1.begin());
    std::fill(marker.begin(), marker.end(), -1);

    std::vector<int> col1(ptr1.back());

    for(int ip = 0, i = 0; ip < np; ++ip) {
        int row_beg = ptr1[ip];
        int row_end = row_beg;

        for(int k = 0; k < block_size; ++k, ++i) {
            for(int j = ptr[i]; j < ptr[i+1]; ++j) {
                int cp = col[j] / block_size;

                if (marker[cp] < row_beg) {
                    marker[cp] = row_end;
                    col1[row_end++] = cp;
                }
            }
        }
    }


    // Transpose pointwise matrix
    int nnz = ptr1.back();

    std::vector<int> ptr2(np + 1, 0);
    std::vector<int> col2(nnz);

    for(int i = 0; i < nnz; ++i)
        ++( ptr2[ col1[i] + 1 ] );

    std::partial_sum(ptr2.begin(), ptr2.end(), ptr2.begin());

    for(int i = 0; i < np; ++i)
        for(int j = ptr1[i]; j < ptr1[i+1]; ++j)
            col2[ptr2[col1[j]]++] = i;

    std::rotate(ptr2.begin(), ptr2.end() - 1, ptr2.end());
    ptr2.front() = 0;

    // Merge both matrices.
    std::fill(marker.begin(), marker.end(), -1);
    pptr.resize(np + 1, 0);

    for(int i = 0; i < np; ++i) {
        for(int j = ptr1[i]; j < ptr1[i+1]; ++j) {
            int c = col1[j];
            if (marker[c] != i) {
                marker[c] = i;
                ++pptr[i + 1];
            }
        }

        for(int j = ptr2[i]; j < ptr2[i+1]; ++j) {
            int c = col2[j];
            if (marker[c] != i) {
                marker[c] = i;
                ++pptr[i + 1];
            }
        }
    }

    std::partial_sum(pptr.begin(), pptr.end(), pptr.begin());
    std::fill(marker.begin(), marker.end(), -1);

    pcol.resize(pptr.back());

    for(int i = 0; i < np; ++i) {
        int row_beg = pptr[i];
        int row_end = row_beg;

        for(int j = ptr1[i]; j < ptr1[i+1]; ++j) {
            int c = col1[j];

            if (marker[c] < row_beg) {
                marker[c] = row_end;
                pcol[row_end++] = c;
            }
        }

        for(int j = ptr2[i]; j < ptr2[i+1]; ++j) {
            int c = col2[j];

            if (marker[c] < row_beg) {
                marker[c] = row_end;
                pcol[row_end++] = c;
            }
        }
    }
}

//---------------------------------------------------------------------------
std::vector<int> pointwise_partition(
        int npart,
        const std::vector<int> &ptr,
        const std::vector<int> &col
        )
{
    int nrows = ptr.size() - 1;

    std::vector<int> part(nrows);

    if (npart == 1) {
        std::fill(part.begin(), part.end(), 0);
    } else {
        int edgecut;

#if defined(METIS_VER_MAJOR) && (METIS_VER_MAJOR >= 5)
        int nconstraints = 1;
        METIS_PartGraphKway(
                &nrows, //nvtxs
                &nconstraints, //ncon -- new
                const_cast<int*>(ptr.data()), //xadj
                const_cast<int*>(col.data()), //adjncy
                NULL, //vwgt
                NULL, //vsize -- new
                NULL, //adjwgt
                &npart,
                NULL,//real t *tpwgts,
                NULL,// real t ubvec
                NULL,
                &edgecut,
                part.data()
                );
#else
        int wgtflag = 0;
        int numflag = 0;
        int options = 0;

        METIS_PartGraphKway(
                &nrows,
                const_cast<int*>(ptr.data()),
                const_cast<int*>(col.data()),
                NULL,
                NULL,
                &wgtflag,
                &numflag,
                &npart,
                &options,
                &edgecut,
                part.data()
                );
#endif
    }

    return part;
}

//---------------------------------------------------------------------------
std::vector<int> partition(
        int n, int nparts, int block_size,
        const std::vector<int> &ptr, const std::vector<int> &col
        )
{
    // Pointwise graph
    std::vector<int> pptr, pcol;
    pointwise_graph(n, block_size, ptr, col, pptr, pcol);

    // Pointwise partition
    std::vector<int> ppart = pointwise_partition(nparts, pptr, pcol);

    std::vector<int> part(n);
    for(int i = 0; i < n; ++i)
        part[i] = ppart[i / block_size];

    return part;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;

    try {
        std::string ifile;
        std::string ofile = "partition.mtx";

        int nparts, block_size;

        po::options_description desc("Options");

        desc.add_options()
            ("help,h", "show help")
            ("input,i",      po::value<std::string>(&ifile)->required(), "Input matrix")
            ("output,o",     po::value<std::string>(&ofile)->default_value(ofile), "Output file")
            (
             "binary,B",
             po::bool_switch()->default_value(false),
             "When specified, treat input files as binary instead of as MatrixMarket. "
            )
            ("nparts,n",     po::value<int>(&nparts)->required(), "Number of parts")
            ("block_size,b", po::value<int>(&block_size)->default_value(1), "Block size")
            ;

        po::positional_options_description pd;
        pd.add("input", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

        size_t rows;
        std::vector<int> ptr, col;
        std::vector<double> val;

        bool binary = vm["binary"].as<bool>();

        if (binary) {
            amgcl::io::read_crs(ifile, rows, ptr, col, val);
        } else {
            size_t cols;
            boost::tie(rows, cols) = amgcl::io::mm_reader(ifile)(ptr, col, val);
            precondition(rows == cols, "Non-square system matrix");
        }

        std::vector<int> part = partition(rows, nparts, block_size, ptr, col);

        if (binary) {
            std::ofstream p(ofile.c_str(), std::ios::binary);

            amgcl::io::write(p, rows);
            amgcl::io::write(p, size_t(1));
            amgcl::io::write(p, part);
        } else {
            amgcl::io::mm_write(ofile, &part[0], part.size());
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
