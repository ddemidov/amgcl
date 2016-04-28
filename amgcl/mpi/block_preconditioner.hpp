#ifndef AMGCL_MPI_BLOCK_PRECONDITIONER_HPP
#define AMGCL_MPI_BLOCK_PRECONDITIONER_HPP

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
 * \file   amgcl/mpi/block_preconditioner.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed solver based on block preconditioning.
 */

#include <vector>

#include <mpi.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/numeric.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/inner_product.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {

template <class Precond, template <class, class> class IterativeSolver>
class block_preconditioner {
    public:
        typedef typename Precond::backend_type Backend;
        typedef IterativeSolver<Backend, mpi::inner_product> Solver;
        typedef typename Backend::params backend_params;

        struct params {
            typename Precond::params precond;
            typename Solver::params  solver;

            params() {}

            params(const boost::property_tree::ptree &p)
                : AMGCL_PARAMS_IMPORT_CHILD(p, precond),
                  AMGCL_PARAMS_IMPORT_CHILD(p, solver)
            {}

            void get(boost::property_tree::ptree &p, const std::string &path) const {
                AMGCL_PARAMS_EXPORT_CHILD(p, path, precond);
                AMGCL_PARAMS_EXPORT_CHILD(p, path, solver);
            }
        };

        typedef typename Backend::value_type value_type;
        typedef typename Backend::matrix     matrix;
        typedef typename Backend::vector     vector;

        template <class Matrix>
        block_preconditioner(
                MPI_Comm mpi_comm,
                const Matrix &Astrip,
                const params &prm = params(),
                const backend_params &bprm = backend_params()
                )
          : comm(mpi_comm), n(backend::rows(Astrip)),
            S(boost::make_shared<Solver>(n, prm.solver, bprm, mpi::inner_product(mpi_comm)))
        {
            typedef backend::crs<value_type> build_matrix;
            typedef typename backend::row_iterator<Matrix>::type row_iterator;

            // Get sizes of each domain in comm.
            std::vector<ptrdiff_t> domain(comm.size + 1, 0);
            MPI_Allgather(&n, 1, datatype<ptrdiff_t>(), &domain[1], 1, datatype<ptrdiff_t>(), comm);
            boost::partial_sum(domain, domain.begin());

            ptrdiff_t loc_beg = domain[comm.rank];
            ptrdiff_t loc_end = domain[comm.rank + 1];

            // Split the matrix into local and remote parts.
            boost::shared_ptr<build_matrix> Aloc = boost::make_shared<build_matrix>();
            boost::shared_ptr<build_matrix> Arem = boost::make_shared<build_matrix>();

            Aloc->nrows = n;
            Aloc->ncols = n;
            Arem->nrows = n;

            Aloc->ptr.resize(n + 1, 0);
            Arem->ptr.resize(n + 1, 0);

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < n; ++i) {
                for(row_iterator a = backend::row_begin(Astrip, i); a; ++a) {
                    ptrdiff_t c = a.col();

                    if (loc_beg <= c && c < loc_end)
                        ++Aloc->ptr[i + 1];
                    else
                        ++Arem->ptr[i + 1];
                }
            }

            boost::partial_sum(Aloc->ptr, Aloc->ptr.begin());
            boost::partial_sum(Arem->ptr, Arem->ptr.begin());

            Aloc->col.resize(Aloc->ptr.back());
            Aloc->val.resize(Aloc->ptr.back());

            Arem->col.resize(Arem->ptr.back());
            Arem->val.resize(Arem->ptr.back());

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < n; ++i) {
                ptrdiff_t loc_head = Aloc->ptr[i];
                ptrdiff_t rem_head = Arem->ptr[i];

                for(row_iterator a = backend::row_begin(Astrip, i); a; ++a) {
                    ptrdiff_t  c = a.col();
                    value_type v = a.value();

                    if (loc_beg <= c && c < loc_end) {
                        Aloc->col[loc_head] = c - loc_beg;
                        Aloc->val[loc_head] = v;
                        ++loc_head;
                    } else {
                        Arem->col[rem_head] = c;
                        Arem->val[rem_head] = v;
                        ++rem_head;
                    }
                }
            }

            P = boost::make_shared<Precond>(Aloc, prm.precond, bprm);
            A = boost::make_shared< distributed_matrix<Backend> >(mpi_comm, P->system_matrix(), Arem, bprm);
        }

        template <class Vec1, class Vec2>
        boost::tuple<size_t, value_type>
        operator()(const Vec1 &rhs, Vec2 &x) const {
            return (*S)(*A, *P, rhs, x);
        }

    private:
        communicator comm;
        ptrdiff_t n;
        boost::shared_ptr<Solver>  S;
        boost::shared_ptr<Precond> P;
        boost::shared_ptr< distributed_matrix<Backend> > A;
};

} // namespace mpi
} // namespace amgcl

#endif
