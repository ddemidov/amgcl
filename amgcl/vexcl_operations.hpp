#ifndef AMGCL_VEXCL_OPERATIONS_HPP
#define AMGCL_VEXCL_OPERATIONS_HPP

namespace amg {

template <typename T>
struct value_type {
    typedef typename T::value_type type;
};

template <typename T>
T inner_prod(const vex::vector<T> &x, const vex::vector<T> &y) {
    static vex::Reductor<T, vex::SUM> sum(vex::StaticContext<>::get().queue());
    return sum(x * y);
}

template <typename T>
T norm(const vex::vector<T> &x) {
    return sqrt( inner_prod(x, x) );
}

template <typename T>
void clear(vex::vector<T> &x) {
    x = 0;
}

}


#endif
