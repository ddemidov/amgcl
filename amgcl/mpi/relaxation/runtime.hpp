#ifndef AMGCL_MPI_RELAXATION_RUNTIME_HPP
#define AMGCL_MPI_RELAXATION_RUNTIME_HPP

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
 * \file   amgcl/mpi/relaxation/runtime.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed memory sparse approximate inverse relaxation scheme.
 */

#include <boost/shared_ptr.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/mpi/relaxation/spai0.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace runtime {
namespace mpi {

template <class Backend>
struct relaxation {
    typedef boost::property_tree::ptree params;
    typedef typename Backend::params    backend_params;

    runtime::relaxation::type r;
    void *handle;

    template <class L, class R>
    relaxation(const amgcl::mpi::distributed_matrix<Backend, L, R> &A,
            params prm, const backend_params &bprm = backend_params())
      : r(prm.get("type", runtime::relaxation::spai0)), handle(0)
    {
        if (!prm.erase("type")) AMGCL_PARAM_MISSING("type");

        switch(r) {

#define AMGCL_RELAX_DISTR(type) \
            case runtime::relaxation::type: \
                handle = static_cast<void*>(new amgcl::mpi::relaxation::type<Backend>(A, prm, bprm)); \
                break

#define AMGCL_RELAX_LOCAL(type) \
            case runtime::relaxation::type: \
                handle = static_cast<void*>(new amgcl::relaxation::type<Backend>(*A.local(), prm, bprm)); \
                break;

            AMGCL_RELAX_DISTR(spai0);
            AMGCL_RELAX_LOCAL(damped_jacobi);
            AMGCL_RELAX_LOCAL(ilu0);
            AMGCL_RELAX_LOCAL(iluk);
            AMGCL_RELAX_LOCAL(ilut);
            AMGCL_RELAX_LOCAL(spai1);
            AMGCL_RELAX_LOCAL(chebyshev);
            AMGCL_RELAX_LOCAL(gauss_seidel);

#undef AMGCL_RELAX_LOCAL
#undef AMGCL_RELAX_DISTR

            default:
                throw std::invalid_argument("Unsupported relaxation type");
        }
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp) const {
        switch(r) {

#define AMGCL_RELAX_DISTR(type) \
            case runtime::relaxation::type: \
                static_cast<const amgcl::mpi::relaxation::type<Backend>*>(handle)->apply_pre(A, rhs, x, tmp); \
                break

#define AMGCL_RELAX_LOCAL_DISTR(type) \
            case runtime::relaxation::type: \
                static_cast<amgcl::relaxation::type<Backend>*>(handle)->apply_pre(A, rhs, x, tmp); \
                break;

#define AMGCL_RELAX_LOCAL_LOCAL(type) \
            case runtime::relaxation::type: \
                static_cast<amgcl::relaxation::type<Backend>*>(handle)->apply_pre(*A.local_backend(), rhs, x, tmp); \
                break;

            AMGCL_RELAX_DISTR(spai0);
            AMGCL_RELAX_LOCAL_DISTR(damped_jacobi);
            AMGCL_RELAX_LOCAL_DISTR(ilu0);
            AMGCL_RELAX_LOCAL_DISTR(iluk);
            AMGCL_RELAX_LOCAL_DISTR(ilut);
            AMGCL_RELAX_LOCAL_DISTR(spai1);
            AMGCL_RELAX_LOCAL_DISTR(chebyshev);
            AMGCL_RELAX_LOCAL_LOCAL(gauss_seidel);

#undef AMGCL_RELAX_LOCAL_LOCAL
#undef AMGCL_RELAX_LOCAL_DISTR
#undef AMGCL_RELAX_DISTR

            default:
                throw std::invalid_argument("Unsupported relaxation type");
        }
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp) const {
        switch(r) {

#define AMGCL_RELAX_DISTR(type) \
            case runtime::relaxation::type: \
                static_cast<const amgcl::mpi::relaxation::type<Backend>*>(handle)->apply_post(A, rhs, x, tmp); \
                break

#define AMGCL_RELAX_LOCAL_DISTR(type) \
            case runtime::relaxation::type: \
                static_cast<amgcl::relaxation::type<Backend>*>(handle)->apply_post(A, rhs, x, tmp); \
                break;

#define AMGCL_RELAX_LOCAL_LOCAL(type) \
            case runtime::relaxation::type: \
                static_cast<amgcl::relaxation::type<Backend>*>(handle)->apply_post(*A.local_backend(), rhs, x, tmp); \
                break;

            AMGCL_RELAX_DISTR(spai0);
            AMGCL_RELAX_LOCAL_DISTR(damped_jacobi);
            AMGCL_RELAX_LOCAL_DISTR(ilu0);
            AMGCL_RELAX_LOCAL_DISTR(iluk);
            AMGCL_RELAX_LOCAL_DISTR(ilut);
            AMGCL_RELAX_LOCAL_DISTR(spai1);
            AMGCL_RELAX_LOCAL_DISTR(chebyshev);
            AMGCL_RELAX_LOCAL_LOCAL(gauss_seidel);

#undef AMGCL_RELAX_LOCAL_LOCAL
#undef AMGCL_RELAX_LOCAL_DISTR
#undef AMGCL_RELAX_DISTR

            default:
                throw std::invalid_argument("Unsupported relaxation type");
        }
    }

    template <class Matrix, class VectorRHS, class VectorX>
    void apply(const Matrix &A, const VectorRHS &rhs, VectorX &x) const {
        switch(r) {

#define AMGCL_RELAX_DISTR(type) \
            case runtime::relaxation::type: \
                static_cast<const amgcl::mpi::relaxation::type<Backend>*>(handle)->apply(A, rhs, x); \
                break

#define AMGCL_RELAX_LOCAL_DISTR(type) \
            case runtime::relaxation::type: \
                static_cast<amgcl::relaxation::type<Backend>*>(handle)->apply(A, rhs, x); \
                break;

#define AMGCL_RELAX_LOCAL_LOCAL(type) \
            case runtime::relaxation::type: \
                static_cast<amgcl::relaxation::type<Backend>*>(handle)->apply(*A.local_backend(), rhs, x); \
                break;

            AMGCL_RELAX_DISTR(spai0);
            AMGCL_RELAX_LOCAL_DISTR(damped_jacobi);
            AMGCL_RELAX_LOCAL_LOCAL(gauss_seidel);
            AMGCL_RELAX_LOCAL_DISTR(ilu0);
            AMGCL_RELAX_LOCAL_DISTR(iluk);
            AMGCL_RELAX_LOCAL_DISTR(ilut);
            AMGCL_RELAX_LOCAL_DISTR(spai1);
            AMGCL_RELAX_LOCAL_DISTR(chebyshev);

#undef AMGCL_RELAX_LOCAL_LOCAL
#undef AMGCL_RELAX_LOCAL_DISTR
#undef AMGCL_RELAX_DISTR

            default:
                throw std::invalid_argument("Unsupported relaxation type");
        }
    }
};

} // namespace mpi
} // namespace runtime
} // namespace amgcl

#endif
