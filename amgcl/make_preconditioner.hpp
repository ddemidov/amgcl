#ifndef AMGCL_MAKE_PRECONDITIONER_HPP
#define AMGCL_MAKE_PRECONDITIONER_HPP

/*
The MIT License

Copyright (c) 2012-2019 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/make_preconditioner.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Pack an amgcl preconditioner and the system matrix it was constructed for.
 */

#include <memory>
#include <type_traits>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {

/// Convenience class that bundles together a preconditioner and the system matrix it was constructed for.
template <class Precond>
struct make_preconditioner {
    typedef typename Precond::backend_type backend_type;

    typedef typename backend_type::value_type value_type;
    typedef typename backend_type::matrix     matrix;
    typedef typename backend_type::vector     vector;
    typedef typename backend_type::params     backend_params;

    typedef typename Precond::params params;

    std::shared_ptr<matrix>  A;
    std::shared_ptr<Precond> P;

    template <class Matrix>
    make_preconditioner(
            const Matrix &M,
            const params &prm = params(),
            const backend_params &bprm = backend_params()
            )
    {
        auto B = std::make_shared<backend::crs<value_type>>(M);
        A = backend_type::copy_matrix(B, bprm);
        P = std::make_shared<Precond>(*B, prm, bprm);
    }

    make_preconditioner(std::shared_ptr<matrix> A, std::shared_ptr<Precond> P)
        : A(A), P(P) {}

    template <class Vec1, class Vec2>
    void apply(const Vec1 &rhs, Vec2 &x) const {
        P->apply(*A, rhs, x);
    }

    template <class Matrix, class Vec1, class Vec2>
    void apply(const Matrix &A, const Vec1 &rhs, Vec2 &x) const {
        P->apply(A, rhs, x);
    }

    size_t bytes() const {
        return backend::bytes(*A) + backend::bytes(*P);
    }

    friend std::ostream& operator<<(std::ostream &os, const make_preconditioner &p) {
        return os << *p.P;
    }

    std::shared_ptr<matrix> system_matrix_ptr() const {
        return A;
    }

    const matrix& system_matrix() const {
        return *A;
    }
};

} // namespace amgcl

#endif
