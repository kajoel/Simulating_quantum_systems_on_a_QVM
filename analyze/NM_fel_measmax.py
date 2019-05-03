# @author=Carl, 30/4
#Imports
import numpy as np
from core.interface import vqe_nelder_mead, hamiltonians_of_size, create_and_convert

def fel_measmax(x, fun, identifier, fun_evals):
    fun_none = []
    vqe = vqe_nelder_mead(fatol=1e-3)
    ansatz_name = identifier[1]
    size = identifier[0]
    for fun_evals_ in fun_evals:
        if len(fun) < fun_evals_:
            print(len(fun))
            raise ValueError(f'fun_evals={fun_evals_} is to big')
        fun_ = fun[:fun_evals_]
        idx = np.argmin(fun_)
        x_min = x[idx]
        mat_idx = identifier[4]
        h = hamiltonians_of_size(size)[0][mat_idx]
        H, qc, ansatz_, _ = create_and_convert(ansatz_name, h)
        fun_none_ = vqe.expectation(ansatz_(x_min), H, samples=None, qc=qc)[0]
        fun_none.append(fun_none_)

    return np.asarray(fun_none)