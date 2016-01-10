#ifndef AMGCL_IO_MM_HPP
#define AMGCL_IO_MM_HPP

/*
The MIT License

Copyright (c) 2012-2015 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   amgcl/io/mm.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Readers for Matrix Market sparse matrices and dense vectors.
 */

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include <boost/tuple/tuple.hpp>

#include <amgcl/util.hpp>

namespace amgcl {
namespace io {

/// Read sparse matrix in matrix market format.
template <typename Idx, typename Val>
boost::tuple<Idx, Idx> mm_read(
        const std::string &fname,
        std::vector<Idx> &ptr,
        std::vector<Idx> &col,
        std::vector<Val> &val
        )
{
    std::ifstream f(fname.c_str());
    precondition(f, "Failed to open file \"" + fname + "\"");

    const std::string format_error = "MatrixMarket format error in \"" + fname + "\"";

    // Read header, skip comments.
    std::string line;
    {
        precondition(std::getline(f, line), format_error);

        std::istringstream is(line);
        std::string banner, mtx, coord, dtype, storage;

        precondition(
                is >> banner >> mtx >> coord >> dtype >> storage,
                format_error);

        precondition(banner  == "%%MatrixMarket", format_error + " (no banner)");
        precondition(mtx     == "matrix",         format_error + " (not a matrix)");
        precondition(coord   == "coordinate",     format_error + " (not a sparse matrix)");
        precondition(dtype   == "real",           format_error + " (not real)");
        precondition(storage == "general",        format_error + " (not general)");

        do {
            precondition(std::getline(f, line), format_error + " (unexpected eof)");
        } while (line[0] == '%');
    }

    // Read sizes
    Idx n, m, nnz;
    {
        std::istringstream is(line);
        precondition(is >> n >> m >> nnz, format_error);
    }

    ptr.clear(); ptr.reserve(n+1);
    col.clear(); col.reserve(nnz);
    val.clear(); val.reserve(nnz);

    for(Idx k = 0, last_i = 0; k < nnz; ++k) {
        precondition(std::getline(f, line), format_error + " (unexpected eof)");
        std::istringstream is(line);

        Idx i, j;
        Val v;

        precondition(is >> i >> j >> v, format_error);

        while(last_i < i) {
            ptr.push_back(col.size());
            last_i++;
        }

        col.push_back(j-1);
        val.push_back(v);
    }

    ptr.push_back(col.size());

    precondition(
            static_cast<Idx>(ptr.size()) == n + 1 && ptr.back() == nnz,
            format_error + " (inconsistent data)");

    return boost::make_tuple(n, m);
}

/// Read dense matrix in matrix market format.
template <typename Val>
boost::tuple<size_t, size_t> mm_read(
        const std::string &fname,
        std::vector<Val> &val
        )
{
    std::ifstream f(fname.c_str());
    precondition(f, "Failed to open file \"" + fname + "\"");

    const std::string format_error = "MatrixMarket format error in \"" + fname + "\"";

    // Read header, skip comments.
    std::string line;
    {
        precondition(std::getline(f, line), format_error);

        std::istringstream is(line);
        std::string banner, mtx, coord, dtype, storage;

        precondition(
                is >> banner >> mtx >> coord >> dtype >> storage,
                format_error);

        precondition(banner  == "%%MatrixMarket", format_error + " (no banner)");
        precondition(mtx     == "matrix",         format_error + " (not a matrix)");
        precondition(coord   == "array",          format_error + " (not a dense array)");
        precondition(dtype   == "real",           format_error + " (not real)");
        precondition(storage == "general",        format_error + " (not general)");

        do {
            precondition(std::getline(f, line), format_error + " (unexpected eof)");
        } while (line[0] == '%');
    }

    // Read sizes
    size_t n, m;
    {
        std::istringstream is(line);
        precondition(is >> n >> m, format_error);
    }

    val.resize(n * m);

    for(size_t j = 0; j < m; ++j) {
        for(size_t i = 0; i < n; ++i) {
            precondition(std::getline(f, line), format_error + " (unexpected eof)");
            std::istringstream is(line);

            Val v;
            precondition(is >> v, format_error);

            val[i * m + j] = v;
        }
    }

    return boost::make_tuple(n, m);
}

} // namespace io
} // namespace amgcl


#endif
