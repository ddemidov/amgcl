#ifndef AMGCL_IO_BINARY_HPP
#define AMGCL_IO_BINARY_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/io/binary.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Convenience functions for binary file I/O.
 */

#include <vector>
#include <fstream>

namespace amgcl {
namespace io {

/// Read single value from a binary file.
template <class T>
bool read(std::ifstream &f, T &val) {
    return f.read((char*)&val, sizeof(T));
}

/// Read vector from a binary file.
template <class T>
bool read(std::ifstream &f, std::vector<T> &vec) {
    return f.read((char*)&vec[0], sizeof(T) * vec.size());
}

/// Write single value to a binary file.
template <class T>
bool write(std::ofstream &f, const T &val) {
    return f.write((char*)&val, sizeof(T));
}

/// Write vector to a binary file.
template <class T>
bool write(std::ofstream &f, const std::vector<T> &vec) {
    return f.write((char*)&vec[0], sizeof(T) * vec.size());
}

} // namespace io
} // namespace amgcl

#endif
