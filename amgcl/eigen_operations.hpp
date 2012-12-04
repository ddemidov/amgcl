#ifndef AMGCL_EIGEN_OPERATIONS_HPP
#define AMGCL_EIGEN_OPERATIONS_HPP

namespace amg {

template <typename T>
struct value_type {
    typedef typename T::Scalar type;
};

template <typename T1, typename T2>
double inner_prod(const Eigen::MatrixBase<T1> &x, const Eigen::MatrixBase<T2> &y) {
    return x.dot(y);
}

template <typename T>
typename T::Scalar norm(const Eigen::MatrixBase<T> &x) {
    return x.norm();
}

template <typename T>
void clear(Eigen::MatrixBase<T> &x) {
    x.setZero();
}

}


#endif
