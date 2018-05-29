#ifndef AMGCL_MPI_COARSENING_LOCAL_HPP
#define AMGCL_MPI_COARSENING_LOCAL_HPP

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
 * \file   amgcl/mpi/coarsening/local.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed memory coarsening based on a local scheme.
 */

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/util.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace coarsening {

template <class Backend, template <class> class LocalBase>
struct local {
    typedef typename LocalBase<Backend>::params params;
    LocalBase<Backend> base;

    local(const params &prm = params()) : base(prm) {}

    boost::tuple<
        boost::shared_ptr< distributed_matrix<Backend> >,
        boost::shared_ptr< distributed_matrix<Backend> >
        >
    transfer_operators(const distributed_matrix<Backend> &A) {
        typedef distributed_matrix<Backend> DM;
        typedef typename Backend::value_type value_type;
        typedef backend::crs<value_type> build_matrix;

        // Use local part of A with local coarsening:
        boost::shared_ptr<build_matrix> P_loc, R_loc;
        boost::tie(P_loc, R_loc) = base.transfer_operators(*A.local());

        boost::shared_ptr<build_matrix> P_rem = boost::make_shared<build_matrix>();
        boost::shared_ptr<build_matrix> R_rem = boost::make_shared<build_matrix>();

        P_rem->set_size(P_loc->nrows, 0, true);
        R_rem->set_size(R_loc->nrows, 0, true);

        return boost::make_tuple(
                boost::make_shared<DM>(A.comm(), P_loc, P_rem),
                boost::make_shared<DM>(A.comm(), R_loc, R_rem)
                );
    }

    boost::shared_ptr< distributed_matrix<Backend> >
    coarse_operator(
            const distributed_matrix<Backend> &A,
            const distributed_matrix<Backend> &P,
            const distributed_matrix<Backend> &R
            ) const
    {
        return base.coarse_operator(A, P, R);
    }
};

} // namespace coarsening
} // namespace mpi
} // namespace amgcl

#endif
