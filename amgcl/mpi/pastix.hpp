#ifndef AMGCL_MPI_PASTIX_HPP
#define AMGCL_MPI_PASTIX_HPP

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
\file   amgcl/mpi/pastix.hpp
\author Denis Demidov <dennis.demidov@gmail.com>
\brief  Wrapper for PaStiX distributed sparse solver.

See http://pastix.gforge.inria.fr
*/

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/irange.hpp>
#include <boost/foreach.hpp>

#include <amgcl/util.hpp>
#include <amgcl/mpi/util.hpp>

extern "C" {
#include <pastix.h>
}

namespace amgcl {
namespace mpi {

template <typename value_type>
class PaStiX {
    public:
        BOOST_STATIC_ASSERT_MSG(
                (boost::is_same<value_type, pastix_float_t>::value),
                "Unsupported value type for PaStiX solver"
                );
        struct params {};

        static int comm_size(int n_global_rows) {
            const int dofs_per_process = 5000;
            return (n_global_rows + dofs_per_process - 1) / dofs_per_process;
        }

        template <class PRng, class CRng, class VRng>
        PaStiX(
                MPI_Comm mpi_comm,
                int n_local_rows,
                const PRng &p_ptr,
                const CRng &p_col,
                const VRng &p_val,
                const params &prm = params()
                )
            : comm(mpi_comm), nrows(n_local_rows), pastix_data(0),
              ptr(boost::begin(p_ptr), boost::end(p_ptr)),
              col(boost::begin(p_col), boost::end(p_col)),
              val(boost::begin(p_val), boost::end(p_val)),
              row(nrows), perm(nrows, 0), invp(nrows, 0)
        {
            std::vector<int> domain(comm.size + 1, 0);
            MPI_Allgather(&nrows, 1, MPI_INT, &domain[1], 1, MPI_INT, comm);
            boost::partial_sum(domain, domain.begin());

            boost::copy(
                    boost::irange(domain[comm.rank], domain[comm.rank + 1]),
                    row.begin()
                    );

            boost::copy(boost::irange(1, nrows + 1), perm.begin());
            boost::copy(boost::irange(1, nrows + 1), invp.begin());

            // PaStiX needs 1-based matrices:
            BOOST_FOREACH(pastix_int_t &p, ptr) ++p;
            BOOST_FOREACH(pastix_int_t &c, col) ++c;
            BOOST_FOREACH(pastix_int_t &r, row) ++r;

            // Initialize parameters with default values:
            iparm[IPARM_MODIFY_PARAMETER] = API_NO;
            call_pastix(API_TASK_INIT, API_TASK_INIT, NULL);

            // Factorize the matrix.
            iparm[IPARM_VERBOSE        ] = API_VERBOSE_NOT;
            iparm[IPARM_RHS_MAKING     ] = API_RHS_B;
            iparm[IPARM_SYM            ] = API_SYM_NO;
            iparm[IPARM_FACTORIZATION  ] = API_FACT_LU;
            iparm[IPARM_TRANSPOSE_SOLVE] = API_YES;
            call_pastix(API_TASK_ORDERING, API_TASK_NUMFACT, NULL);
        }

        ~PaStiX() {
            call_pastix(API_TASK_CLEAN, API_TASK_CLEAN, NULL);
        }

        template <class Vec1, class Vec2>
        void operator()(const Vec1 &rhs, Vec2 &x) const {
            boost::copy(rhs, &x[0]);
            call_pastix(API_TASK_SOLVE, API_TASK_SOLVE, &x[0]);
        }
    private:
        amgcl::mpi::communicator comm;

        int nrows;

        // Pastix internal data.
        mutable pastix_data_t *pastix_data;

        // Pastix parameters
        mutable pastix_int_t   iparm[IPARM_SIZE];
        mutable double         dparm[DPARM_SIZE];

        std::vector<pastix_int_t>   ptr;
        std::vector<pastix_int_t>   col;
        std::vector<pastix_float_t> val;

        // Local to global mapping
        std::vector<pastix_int_t> row;

        // Permutation arrays
        std::vector<pastix_int_t> perm, invp;

        void call_pastix(int beg, int end, pastix_float_t *x) const {
            iparm[IPARM_START_TASK] = beg;
            iparm[IPARM_END_TASK  ] = end;

            dpastix(&pastix_data, comm, nrows,
                    const_cast<pastix_int_t*  >(ptr.data()),
                    const_cast<pastix_int_t*  >(col.data()),
                    const_cast<pastix_float_t*>(val.data()),
                    const_cast<pastix_int_t*  >(row.data()),
                    const_cast<pastix_int_t*  >(perm.data()),
                    const_cast<pastix_int_t*  >(invp.data()),
                    x, 1, iparm, dparm
                   );
        }
};

}
}

#endif
