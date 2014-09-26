import numpy
import pyamgcl_ext
from pyamgcl_ext import coarsening, relaxation, solver_type
from scipy.sparse.linalg import LinearOperator

class make_solver:
    def __init__(self,
            A,
            coarsening=pyamgcl_ext.coarsening.smoothed_aggregation,
            relaxation=pyamgcl_ext.relaxation.spai0,
            solver=pyamgcl_ext.solver_type.bicgstabl,
            prm={}
            ):
        Acsr = A.tocsr()

        self.S = pyamgcl_ext.make_solver(
                coarsening, relaxation, solver, prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64)
                )

    def __str__(self):
        return self.S.__str__()

    def __call__(self, rhs):
        return self.S(rhs.astype(numpy.float64))

class make_preconditioner(LinearOperator):
    def __init__(self,
            A,
            coarsening=pyamgcl_ext.coarsening.smoothed_aggregation,
            relaxation=pyamgcl_ext.relaxation.spai0,
            prm={}
            ):
        Acsr = A.tocsr()

        self.P = pyamgcl_ext.make_preconditioner(
                coarsening, relaxation, prm,
                Acsr.indptr.astype(numpy.int32),
                Acsr.indices.astype(numpy.int32),
                Acsr.data.astype(numpy.float64)
                )

        super(make_preconditioner, self).__init__(A.shape, self.P)

    def __str__(self):
        return self.P.__str__()

    def __call__(self, rhs):
        return self.P(rhs.astype(numpy.float64))

