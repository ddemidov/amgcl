#ifndef AMGCL_MPI_COARSENING_SMOOTHED_AGGREGATION_HPP
#define AMGCL_MPI_COARSENING_SMOOTHED_AGGREGATION_HPP

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
 * \file   amgcl/mpi/coarsening/smoothed_aggregation.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed memory smoothed aggregation coarsening scheme.
 */

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/util.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace coarsening {

template <class Backend>
struct smoothed_aggregation {
    typedef typename amgcl::coarsening::smoothed_aggregation<Backend> Base;
    typedef typename Base::params params;
    params prm;

    smoothed_aggregation(const params &prm = params()) : prm(prm) {}

    template <class LM, class RM>
    boost::tuple<
        boost::shared_ptr< distributed_matrix<Backend, LM, RM> >,
        boost::shared_ptr< distributed_matrix<Backend, LM, RM> >
        >
    transfer_operators(const distributed_matrix<Backend, LM, RM> &A) {
        typedef typename Base::Aggregates    Aggregates;
        typedef distributed_matrix<Backend, LM, RM>        DM;
        typedef typename Backend::value_type               value_type;
        typedef backend::crs<value_type>                   build_matrix;
        typedef typename math::scalar_of<value_type>::type scalar_type;

        // Use local part of A with to create tentative prolongation operator:
        build_matrix &A_loc = *A.local();
        build_matrix &A_rem = *A.remote();
        const ptrdiff_t n = A_loc.nrows;

        AMGCL_TIC("aggregates");
        Aggregates aggr(A_loc, prm.aggr, prm.nullspace.cols);
        prm.aggr.eps_strong *= 0.5;
        AMGCL_TOC("aggregates");

        AMGCL_TIC("interpolation");
        boost::shared_ptr<build_matrix> P_tent = amgcl::coarsening::tentative_prolongation<build_matrix>(
                n, aggr.count, aggr.id, prm.nullspace, prm.aggr.block_size);

        boost::shared_ptr<build_matrix> P_rem = boost::make_shared<build_matrix>();
        P_rem->set_size(n, 0, true);

        scalar_type omega = prm.relax;
        if (prm.estimate_spectral_radius) {
            omega *= static_cast<scalar_type>(4.0/3) / Base::spectral_radius(A_loc, prm.power_iters);
        } else {
            omega *= static_cast<scalar_type>(2.0/3);
        }

        // Temporarily substitute nonzeros of A with those of the filtered matrix,
        // and use it to obtain the smoothed prolongation operator:
        boost::shared_ptr<DM> P;
        AMGCL_TIC("smoothing");
        {
            backend::numa_vector<value_type> Af_loc(A_loc.nnz, false);
            backend::numa_vector<value_type> Af_rem(A_rem.nnz, false);

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < n; ++i) {
                // find diagonal value:
                value_type D = math::zero<value_type>();

                for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
                    if (A_loc.col[j] == i || !aggr.strong_connection[j])
                        D += A_loc.val[j];
                }

                D = -omega * math::inverse(D);

                for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
                    ptrdiff_t c = A_loc.col[j];

                    if (c != i && !aggr.strong_connection[j]) {
                        Af_loc[j] = math::zero<value_type>();
                        continue;
                    }

                    if (c == i) {
                        Af_loc[j] = (1 - omega) * math::identity<value_type>();
                    } else {
                        Af_loc[j] = D * A_loc.val[j];
                    }

                }

                for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j) {
                    Af_rem[j] = D * A_rem.val[j];
                }
            }

            value_type *A_loc_val = A_loc.val;
            value_type *A_rem_val = A_rem.val;

            A_loc.val = Af_loc.data();
            A_rem.val = Af_rem.data();

            P = product(A, DM(A.comm(), P_tent, P_rem, A.backend_prm()));

            A_loc.val = A_loc_val;
            A_rem.val = A_rem_val;
        }
        AMGCL_TOC("smoothing");
        AMGCL_TOC("interpolation");

        return boost::make_tuple(P, transpose(*P));
    }

    template <class LM, class RM>
    boost::shared_ptr< distributed_matrix<Backend, LM, RM> >
    coarse_operator(
            const distributed_matrix<Backend, LM, RM> &A,
            const distributed_matrix<Backend, LM, RM> &P,
            const distributed_matrix<Backend, LM, RM> &R
            ) const
    {
        return amgcl::coarsening::detail::galerkin(A, P, R);
    }
};

} // namespace coarsening
} // namespace mpi
} // namespace amgcl

#endif
