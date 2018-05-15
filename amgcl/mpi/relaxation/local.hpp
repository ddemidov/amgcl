#ifndef AMGCL_MPI_RELAXATION_LOCAL_HPP
#define AMGCL_MPI_RELAXATION_LOCAL_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/mpi/relaxation/local.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed memory relaxation based on local matrix subblock.
 */

#include <boost/shared_ptr.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace relaxation {

template <class Backend, template <class> class LocalBase>
struct local {
    typedef typename Backend::value_type        value_type;
    typedef typename LocalBase<Backend>::params params;
    typedef typename Backend::params            backend_params;
    
    LocalBase<Backend> base;

    template <class L, class R>
    local(const distributed_matrix<Backend, L, R> &A,
            const params &prm, const backend_params &bprm = backend_params()
         ) : base(*A.local(), prm, bprm) { }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp) const {
        base.apply_pre(*A.local_backend(), rhs, x, tmp);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp) const {
        base.apply_post(*A.local_backend(), rhs, x, tmp);
    }

    template <class Matrix, class VectorRHS, class VectorX>
    void apply(const Matrix &A, const VectorRHS &rhs, VectorX &x) const {
        base.apply(*A.local_backend(), rhs, x);
    }
};

} // namespace relaxation
} // namespace mpi
} // namespace amgcl

#endif
