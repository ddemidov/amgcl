#ifndef AMGCL_AMGCL_HPP
#define AMGCL_AMGCL_HPP

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
 * \file   amgcl/amgcl.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Generic algebraic multigrid framework.
 */

#include <amgcl/builder.hpp>
#include <amgcl/solver.hpp>

namespace amgcl {

template <class Builder, class Solver>
Solver make_solver(
        typename Builder::matrix const &A,
        typename Builder::params const &build_prm,
        typename Solver::params  const &solve_prm
        )
{
    return Solver(Builder(A, build_prm), solve_prm);
}

} // namespace amgcl

#endif
