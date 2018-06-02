#ifndef EXAMPLES_MPI_DOMAIN_PARTITION_HPP
#define EXAMPLES_MPI_DOMAIN_PARTITION_HPP

#include <vector>
#include <utility>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/adapted/boost_array.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

BOOST_GEOMETRY_REGISTER_BOOST_ARRAY_CS(cs::cartesian)

template <int NDIM>
class domain_partition {
    public:
        typedef boost::array<ptrdiff_t, NDIM>      point;
        typedef boost::geometry::model::box<point> box;
        typedef std::pair<box, int>                process;

        domain_partition(point lo, point hi, int num_processes) {
            split(box(lo, hi), num_processes);

            for(int i = 0; i < num_processes; ++i)
                rtree.insert( std::make_pair(subdomains[i], i) );
        }

        std::pair<int, ptrdiff_t> index(point p) const {
            namespace bgi = boost::geometry::index;

            for(const process &v : rtree | bgi::adaptors::queried(bgi::intersects(p)) )
            {
                return std::make_pair(v.second, local_index(v.first, p));
            }

            // Unreachable:
            return std::make_pair(0, 0l);
        }

        size_t size(size_t process) const {
            if (process >= subdomains.size()) return 0;

            point lo = subdomains[process].min_corner();
            point hi = subdomains[process].max_corner();

            size_t v = 1;

            for(int i = 0; i < NDIM; ++i)
                v *= hi[i] - lo[i] + 1;

            return v;
        }

        box domain(size_t process) const {
            if (process < subdomains.size())
                return subdomains[process];
            else {
                point lo;
                point hi;
                for(int i = 0; i < NDIM; ++i) {
                    lo[i] = 0;
                    hi[i] = -1;
                }
                return box(lo, hi);
            }
        }
    private:
        std::vector<box> subdomains;

        boost::geometry::index::rtree<
            process,
            boost::geometry::index::quadratic<16>
            > rtree;

        static ptrdiff_t local_index(box domain, point p) {
            point lo = domain.min_corner();
            point hi = domain.max_corner();

            ptrdiff_t stride = 1, idx = 0;
            for(int i = 0; i < NDIM; ++i) {
                idx += (p[i] - lo[i]) * stride;
                stride *= hi[i] - lo[i] + 1;
            }

            return idx;
        }

        void split(box domain, int np) {
            if (np == 1) {
                subdomains.push_back(domain);
                return;
            }

            point lo = domain.min_corner();
            point hi = domain.max_corner();

            // Get longest dimension of the domain
            int wd = 0;
            for(int i = 1; i < NDIM; ++i)
                if (hi[i] - lo[i] > hi[wd] - lo[wd]) wd = i;

            ptrdiff_t mid = lo[wd] + (hi[wd] - lo[wd]) * (np / 2) / np;

            box sd1 = domain;
            box sd2 = domain;

            sd1.max_corner()[wd] = mid;
            sd2.min_corner()[wd] = mid + 1;

            split(sd1, np / 2);
            split(sd2, np - np / 2);
        }
};

#endif
