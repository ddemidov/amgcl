#ifndef AMGCL_COARSENING_RUGE_STUBEN_HPP
#define AMGCL_COARSENING_RUGE_STUBEN_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/coarsening/ruge_stuben.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Ruge-Stuben coarsening with direct interpolation.
 */

#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace coarsening {

struct ruge_stuben {
    struct params {
        float eps_strong;
        float eps_trunc;
        bool  do_trunc;

        params() : eps_strong(0.25f), eps_trunc(0.2f), trunc_int(true) {}
    };
};

} // namespace coarsening
} // namespace amgcl

#endif
