#ifndef AMGCL_COARSENING_PLAIN_AGGREGATES_HPP
#define AMGCL_COARSENING_PLAIN_AGGREGATES_HPP

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
 * \file   amgcl/coarsening/plain_aggregates.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Plain aggregation.
 */

#include <vector>
#include <boost/foreach.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace coarsening {

struct plain_aggregates {
    static const long undefined = -1;
    static const long removed   = -2;

    size_t count;

    std::vector<char> strong_connection;
    std::vector<long> id;

    template <class Matrix>
    plain_aggregates(const Matrix &A, float eps_strong)
        : count(0),
          strong_connection( backend::nonzeros(A) ),
          id( backend::rows(A) )
    {
        typedef typename backend::value_type<Matrix>::type V;
        V eps_squared = eps_strong * eps_strong;

        const size_t n = rows(A);

        /* 1. Get strong connections */
        std::vector<V> dia = diagonal(A);
#pragma omp parallel for
        for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            V eps_dia_i = eps_squared * dia[i];

            for(long j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                long c = A.col[j];
                V    v = A.val[j];

                strong_connection[j] = (c != i) && (v * v > eps_dia_i * dia[c]);
            }
        }

        /* 2. Get aggregate ids */

        // Remove lonely nodes.
        size_t max_neib = 0;
        for(size_t i = 0; i < n; ++i) {
            long j = A.ptr[i], e = A.ptr[i+1];
            max_neib = std::max<size_t>(max_neib, e - j);

            long state = removed;
            for(; j < e; ++j)
                if (strong_connection[j]) {
                    state = undefined;
                    break;
                }

            id[i] = state;
        }

        std::vector<long> neib;
        neib.reserve(max_neib);

        // Perform plain aggregation
        for(size_t i = 0; i < n; ++i) {
            if (id[i] != undefined) continue;

            // The point is not adjacent to a core of any previous aggregate:
            // so its a seed of a new aggregate.
            long cur_id = count++;
            id[i] = cur_id;

            // Include its neighbors as well.
            neib.clear();
            for(long j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                long c = A.col[j];
                if (strong_connection[j] && id[c] != removed) {
                    id[c] = cur_id;
                    neib.push_back(c);
                }
            }

            // Temporarily mark undefined points adjacent to the new aggregate
            // as members of the aggregate.
            // If nobody claims them later, they will stay here.
            BOOST_FOREACH(long c, neib) {
                for(long j = A.ptr[c], e = A.ptr[c+1]; j < e; ++j) {
                    long cc = A.col[j];
                    if (strong_connection[j] && id[cc] == undefined)
                        id[cc] = cur_id;
                }
            }
        }
    }
};

} // namespace coarsening
} // namespace amgcl

#endif
