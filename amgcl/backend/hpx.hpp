#ifndef AMGCL_BACKEND_HPX_HPP
#define AMGCL_BACKEND_HPX_HPP

#include <vector>
#include <mutex>

#include <hpx/hpx.hpp>
#include <hpx/include/lcos.hpp>

#include <boost/algorithm/minmax.hpp>

#include <amgcl/util.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace backend {

// The matrix is a thin wrapper on top of builtin::crs<>.
template <typename real>
struct hpx_matrix {
    typedef real      value_type;
    typedef ptrdiff_t index_type;

    typedef crs<value_type, index_type> Base;

    // Base matrix is stored in shared_ptr<> to reduce the overhead
    // of data transfer from builtin datatypes (used for AMG setup) to the
    // backend datatypes.
    boost::shared_ptr<Base> base;

    // For each of the output segments y[i] in y = A * x it stores a range of
    // segments in x that y[i] depends on.
    std::vector<std::tuple<index_type, index_type>> xrange;

    // Creates the matrix from builtin datatype, sets up xrange.
    hpx_matrix(boost::shared_ptr<Base> A, int grain_size) : base(A)
    {
        index_type n    = backend::rows(*A);
        index_type nseg = (n + grain_size - 1) / grain_size;

        xrange.reserve(nseg);

        for(index_type i = 0; i < n; i += grain_size) {
            index_type beg = A->ptr[i];
            index_type end = A->ptr[std::min<index_type>(i + grain_size, n)];

            auto mm = std::minmax_element(
                    base->col.begin() + beg, A->col.begin() + end
                    );

            xrange.push_back( std::make_tuple(
                        *std::get<0>(mm) / grain_size,
                        *std::get<1>(mm) / grain_size + 1
                        ));
        }
    }
};

// The vector is a wrapper on top of std::vector.
// The vector is assumed to consist of continuous segments of grain_size
// elements. A vector of shared_futures corresponding to each of the
// segments is stored along the data vector.
template < typename real >
struct hpx_vector {
    typedef real value_type;

    typedef std::vector<real> Base;

    // Segments stored in a continuous array.
    // The base vector is stored with shared_ptr for the same reason as with
    // hpx_matrix above: to reduce the overhead of data transfer.
    boost::shared_ptr<Base> vec;

    // Futures associated with each segment:
    mutable std::vector<hpx::shared_future<void>> fut;

    int nseg;        // Number of segments in the vector
    int grain_size;  // Segment size.

    hpx_vector(size_t n, int grain_size)
        : vec( boost::make_shared<Base>(n) ),
          nseg( (n + grain_size - 1) / grain_size ),
          grain_size( grain_size )
    {
        init_futures();
    }

    hpx_vector(boost::shared_ptr<Base> o, int grain_size)
        : vec(o),
          nseg( (o->size() + grain_size - 1) / grain_size ),
          grain_size( grain_size )
    {
        init_futures();
    }

    void init_futures() {
        fut.reserve(nseg);
        for(ptrdiff_t i = 0; i < nseg; ++i)
            fut.push_back(hpx::make_ready_future());
    }

    size_t size() const { return vec->size(); }

    real operator[](size_t i) const {
        return (*vec)[i];
    }

    real& operator[](size_t i) {
        return (*vec)[i];
    }
};

/// HPX backend
/**
 * This is a backend that is based on HPX -- a general purpose C++ runtime
 * system for parallel and distributed applications of any scale
 * http://stellar-group.org/libraries/hpx.
 */
template <typename real>
struct HPX {
    typedef real      value_type;
    typedef ptrdiff_t index_type;

    struct provides_row_iterator : boost::false_type {};

    struct params {
        /// Number of vector elements in a single segment.
        int grain_size;

        params() : grain_size(4096) {}

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_VALUE(p, grain_size)
        {}
    };

    typedef hpx_matrix<value_type>         matrix;
    typedef hpx_vector<value_type>         vector;

    struct direct_solver : public solver::skyline_lu<value_type> {
        typedef solver::skyline_lu<value_type> Base;
        typedef typename Base::params params;

        template <class Matrix>
        direct_solver(const Matrix &A, const params &prm = params())
            : Base(A, prm)
        {}

        struct call_base {
            const Base *base;
            real *fptr;
            real *xptr;

            template <class T>
            void operator()(T&&) const {
                (*base)(fptr, xptr);
            }
        };

        void operator()(const vector &rhs, vector &x) const {
            real *fptr = rhs.vec->data();
            real *xptr = x.vec->data();

            using hpx::lcos::local::dataflow;

            hpx::shared_future<void> solve = dataflow(
                    hpx::launch::async,
                    call_base{this, fptr, xptr},
                    hpx::when_all(rhs.fut)
                    );

            for(auto f = x.fut.begin(); f != x.fut.end(); ++f)
                *f = solve;

        }
    };

    static std::string name() { return "HPX"; }

    /// Copy matrix.
    static boost::shared_ptr<matrix>
    copy_matrix(boost::shared_ptr<typename matrix::Base> A, const params &p)
    {
        return boost::make_shared<matrix>(A, p.grain_size);
    }

    /// Copy vector to builtin backend.
    static boost::shared_ptr<vector>
    copy_vector(const typename vector::Base &x, const params &p)
    {
        return boost::make_shared<vector>(
                boost::make_shared<typename vector::Base>(x), p.grain_size
                );
    }

    /// Copy vector to builtin backend.
    static boost::shared_ptr<vector>
    copy_vector(boost::shared_ptr<typename vector::Base> x, const params &p)
    {
        return boost::make_shared<vector>(x, p.grain_size);
    }

    /// Create vector of the specified size.
    static boost::shared_ptr<vector>
    create_vector(size_t size, const params &p)
    {
        return boost::make_shared<vector>(size, p.grain_size);
    }

    /// Create direct solver for coarse level
    static boost::shared_ptr<direct_solver>
    create_solver(boost::shared_ptr<typename matrix::Base> A, const params&) {
        return boost::make_shared<direct_solver>(*A);
    }
};

//---------------------------------------------------------------------------
// Backend interface implementation
//---------------------------------------------------------------------------
template < typename real >
struct rows_impl< hpx_matrix<real> > {
    static size_t get(const hpx_matrix<real> &A) {
        return backend::rows(*A.base);
    }
};

template < typename real >
struct cols_impl< hpx_matrix<real> > {
    static size_t get(const hpx_matrix<real> &A) {
        return backend::cols(*A.base);
    }
};

template < typename real >
struct nonzeros_impl< hpx_matrix<real> > {
    static size_t get(const hpx_matrix<real> &A) {
        return backend::nonzeros(*A.base);
    }
};

template < typename real >
struct spmv_impl<
    hpx_matrix<real>,
    hpx_vector<real>,
    hpx_vector<real>
    >
{
    typedef hpx_matrix<real> matrix;
    typedef hpx_vector<real> vector;

    struct process_ab {
        real  alpha;
        const crs<real,ptrdiff_t> *A;
        const real *xptr;
        real  beta;
        real *yptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i) {
                real sum = 0;
                for(auto a = A->row_begin(i); a; ++a)
                    sum += a.value() * xptr[a.col()];
                yptr[i] = alpha * sum + beta * yptr[i];
            }
        }
    };

    struct process_a {
        real  alpha;
        const crs<real,ptrdiff_t> *A;
        const real *xptr;
        real *yptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i) {
                real sum = 0;
                for(auto a = A->row_begin(i); a; ++a)
                    sum += a.value() * xptr[a.col()];
                yptr[i] = alpha * sum;
            }
        }
    };

    struct ignore {
        template <class T>
        void operator()(T&&) const {}
    };

    static void apply(real alpha, const matrix &A, const vector &x,
            real beta, vector &y)
    {
        real *xptr = x.vec->data();
        real *yptr = y.vec->data();

        auto Abase = A.base.get();

        using hpx::lcos::local::dataflow;

        if (beta) {
            // y = alpha * A * x + beta * y
            for(ptrdiff_t seg = 0; seg < y.nseg; ++seg) {
                ptrdiff_t beg = seg * y.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + y.grain_size, y.size());

                y.fut[seg] = dataflow(hpx::launch::async,
                        process_ab{alpha, Abase, xptr, beta, yptr, beg, end},
                        hpx::when_all(
                            y.fut[seg],
                            hpx::when_all(
                                x.fut.begin() + std::get<0>(A.xrange[seg]),
                                x.fut.begin() + std::get<1>(A.xrange[seg])
                                )
                            )
                        );
            }
        } else {
            // y = alpha * A * x
            for(ptrdiff_t seg = 0; seg < y.nseg; ++seg) {
                ptrdiff_t beg = seg * y.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + y.grain_size, y.size());

                y.fut[seg] = dataflow(hpx::launch::async,
                        process_a{alpha, Abase, xptr, yptr, beg, end},
                        hpx::when_all(
                            x.fut.begin() + std::get<0>(A.xrange[seg]),
                            x.fut.begin() + std::get<1>(A.xrange[seg])
                            )
                        );
            }
        }

        // Do not update x until y is ready.
        hpx::shared_future<void> wait_for_it = dataflow(hpx::launch::async,
                ignore(), hpx::when_all(y.fut));
        for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) x.fut[seg] = wait_for_it;
    }
};

template < typename real >
struct residual_impl<
    hpx_matrix<real>,
    hpx_vector<real>,
    hpx_vector<real>,
    hpx_vector<real>
    >
{
    typedef hpx_matrix<real> matrix;
    typedef hpx_vector<real> vector;

    struct process {
        real *fptr;
        const crs<real,ptrdiff_t> *A;
        real *xptr;
        real *rptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i) {
                real sum = fptr[i];
                for(auto a = A->row_begin(i); a; ++a)
                    sum -= a.value() * xptr[a.col()];
                rptr[i] = sum;
            }
        }
    };

    struct ignore {
        template <class T>
        void operator()(T&&) const {}
    };

    static void apply(const vector &f, const matrix &A, const vector &x,
            vector &r)
    {
        real *xptr = x.vec->data();
        real *fptr = f.vec->data();
        real *rptr = r.vec->data();

        auto Abase = A.base.get();

        using hpx::lcos::local::dataflow;

        for(ptrdiff_t seg = 0; seg < f.nseg; ++seg) {
            ptrdiff_t beg = seg * f.grain_size;
            ptrdiff_t end = std::min<ptrdiff_t>(beg + f.grain_size, f.size());

            r.fut[seg] = dataflow(hpx::launch::async,
                    process{fptr, Abase, xptr, rptr, beg, end},
                    hpx::when_all(
                        f.fut[seg],
                        hpx::when_all(
                            x.fut.begin() + std::get<0>(A.xrange[seg]),
                            x.fut.begin() + std::get<1>(A.xrange[seg])
                            )
                        )
                    );
        }

        // Do not update x until r is ready.
        hpx::shared_future<void> wait_for_it = dataflow(hpx::launch::async,
                ignore(), hpx::when_all(r.fut));
        for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) x.fut[seg] = wait_for_it;
    }
};

template < typename real >
struct clear_impl<
    hpx_vector<real>
    >
{
    typedef hpx_vector<real> vector;

    static void apply(vector &x)
    {
        real *xptr = x.vec->data();

        using hpx::lcos::local::dataflow;

        for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
            ptrdiff_t beg = seg * x.grain_size;
            ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

            x.fut[seg] = dataflow(hpx::launch::async,
                    [xptr, beg, end](const hpx::shared_future<void>&) {
                        for(ptrdiff_t i = beg; i < end; ++i) xptr[i] = 0;
                    },
                    x.fut[seg]
                    );
        }
    }
};

template < typename real >
struct copy_impl<
    hpx_vector<real>,
    hpx_vector<real>
    >
{
    typedef hpx_vector<real> vector;

    static void apply(const vector &x, vector &y)
    {
        real *xptr = x.vec->data();
        real *yptr = y.vec->data();

        using hpx::lcos::local::dataflow;

        for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
            ptrdiff_t beg = seg * x.grain_size;
            ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

            y.fut[seg] = dataflow(hpx::launch::async,
                    [xptr, yptr, beg, end](const hpx::shared_future<void>&) {
                        for(ptrdiff_t i = beg; i < end; ++i)
                            yptr[i] = xptr[i];
                    },
                    x.fut[seg]
                    );
        }
    }
};

template < typename real >
struct copy_to_backend_impl<
    hpx_vector<real>
    >
{
    typedef hpx_vector<real> vector;

    static void apply(const std::vector<real> &x, vector &y)
    {
        real *xptr = x.data();
        real *yptr = y.vec->data();

        for(ptrdiff_t seg = 0; seg < y.nseg; ++seg) {
            ptrdiff_t beg = seg * y.grain_size;
            ptrdiff_t end = std::min<ptrdiff_t>(beg + y.grain_size, y.size());

            y.fut[seg] = hpx::async(
                    [xptr, yptr, beg, end]() {
                        for(ptrdiff_t i = beg; i < end; ++i)
                            yptr[i] = xptr[i];
                    });
        }
    }
};

template < typename real >
struct inner_product_impl<
    hpx_vector<real>,
    hpx_vector<real>
    >
{
    typedef hpx_vector<real> vector;

    struct process {
        std::mutex &mx;
        real &tot;
        real *xptr;
        real *yptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            real sum = 0;

            for(ptrdiff_t i = beg; i < end; ++i)
                sum += xptr[i] * yptr[i];

            {
                std::unique_lock<std::mutex> lock(mx);
                tot += sum;
            }
        }
    };

    static real get(const vector &x, const vector &y)
    {
        std::mutex mx;
        real tot = 0;

        real *xptr = x.vec->data();
        real *yptr = y.vec->data();

        using hpx::lcos::local::dataflow;

        for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
            ptrdiff_t beg = seg * x.grain_size;
            ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

            x.fut[seg] = dataflow(hpx::launch::async,
                    process{mx, tot, xptr, yptr, beg, end},
                    hpx::when_all(x.fut[seg], y.fut[seg])
                    );
        }

        hpx::wait_all(x.fut);

        return tot;
    }
};

template < typename real >
struct axpby_impl<
    hpx_vector<real>,
    hpx_vector<real>
    >
{
    typedef hpx_vector<real> vector;

    struct process_ab {
        typedef void result_type;

        real  a;
        real *xptr;
        real  b;
        real *yptr;
        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i)
                yptr[i] = a * xptr[i] + b * yptr[i];
        }
    };

    struct process_a {
        typedef void result_type;

        real  a;
        real *xptr;
        real *yptr;
        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i)
                yptr[i] = a * xptr[i];
        }
    };

    static void apply(real a, const vector &x, real b, vector &y)
    {
        real *xptr = x.vec->data();
        real *yptr = y.vec->data();

        using hpx::lcos::local::dataflow;

        if (b) {
            // y = a * x + b * y;
            for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
                ptrdiff_t beg = seg * x.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

                y.fut[seg] = dataflow(hpx::launch::async,
                        process_ab{a, xptr, b, yptr, beg, end},
                        hpx::when_all(x.fut[seg], y.fut[seg])
                        );
            }
        } else {
            // y = a * x;
            for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
                ptrdiff_t beg = seg * x.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

                y.fut[seg] = dataflow(hpx::launch::async,
                        process_a{a, xptr, yptr, beg, end},
                        x.fut[seg]
                        );
            }
        }
    }
};

template < typename real >
struct axpbypcz_impl<
    hpx_vector<real>,
    hpx_vector<real>,
    hpx_vector<real>
    >
{
    typedef hpx_vector<real> vector;

    struct process_abc {
        real  a;
        real *xptr;
        real  b;
        real *yptr;
        real  c;
        real *zptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i)
                zptr[i] = a * xptr[i] + b * yptr[i] + c * zptr[i];
        }
    };

    struct process_ab {
        real  a;
        real *xptr;
        real  b;
        real *yptr;
        real *zptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i)
                zptr[i] = a * xptr[i] + b * yptr[i];
        }
    };

    static void apply(
            real a, const vector &x,
            real b, const vector &y,
            real c,       vector &z
            )
    {
        real *xptr = x.vec->data();
        real *yptr = y.vec->data();
        real *zptr = z.vec->data();

        using hpx::lcos::local::dataflow;

        if (c) {
            //z = a * x + b * y + c * z;
            for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
                ptrdiff_t beg = seg * x.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

                z.fut[seg] = dataflow(hpx::launch::async,
                        process_abc{a, xptr, b, yptr, c, zptr, beg, end},
                        hpx::when_all(x.fut[seg], y.fut[seg], z.fut[seg])
                        );
            }
        } else {
            //z = a * x + b * y;
            for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
                ptrdiff_t beg = seg * x.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

                z.fut[seg] = dataflow(hpx::launch::async,
                        process_ab{a, xptr, b, yptr, zptr, beg, end},
                        hpx::when_all(x.fut[seg], y.fut[seg])
                        );
            }
        }
    }
};

template < typename real >
struct vmul_impl<
    hpx_vector<real>,
    hpx_vector<real>,
    hpx_vector<real>
    >
{
    typedef hpx_vector<real> vector;

    struct process_ab {
        real  a;
        real *xptr;
        real *yptr;
        real  b;
        real *zptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i)
                zptr[i] = a * xptr[i] * yptr[i] + b * zptr[i];
        }
    };

    struct process_a {
        real  a;
        real *xptr;
        real *yptr;
        real *zptr;

        ptrdiff_t beg;
        ptrdiff_t end;

        template <class T>
        void operator()(T&&) const {
            for(ptrdiff_t i = beg; i < end; ++i)
                zptr[i] = a * xptr[i] * yptr[i];
        }
    };

    static void apply(real a, const vector &x, const vector &y, real b, vector &z)
    {
        real *xptr = x.vec->data();
        real *yptr = y.vec->data();
        real *zptr = z.vec->data();

        using hpx::lcos::local::dataflow;

        if (b) {
            //z = a * x * y + b * z;
            for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
                ptrdiff_t beg = seg * x.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

                z.fut[seg] = dataflow(hpx::launch::async,
                        process_ab{a, xptr, yptr, b, zptr, beg, end},
                        hpx::when_all(x.fut[seg], y.fut[seg], z.fut[seg])
                        );
            }
        } else {
            //z = a * x * y;
            for(ptrdiff_t seg = 0; seg < x.nseg; ++seg) {
                ptrdiff_t beg = seg * x.grain_size;
                ptrdiff_t end = std::min<ptrdiff_t>(beg + x.grain_size, x.size());

                z.fut[seg] = dataflow(hpx::launch::async,
                        process_a{a, xptr, yptr, zptr, beg, end},
                        hpx::when_all(x.fut[seg], y.fut[seg])
                        );
            }
        }
    }
};

} // namespace backend
} // namespace amgcl

#endif
