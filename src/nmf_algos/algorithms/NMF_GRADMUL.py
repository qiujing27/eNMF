import os
import time
import copy
import numpy as np
import numpy.linalg as LA
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.algo_utils import calculate_obj_NMF
from nmf_algos.NMF_base import NMFBase


def findbar(X, Xgrad, sigma):
    assert X.shape == Xgrad.shape
    Xbar = copy.deepcopy(X)
    negative_mask = (Xgrad < 0) * 1.0
    Xbar = negative_mask * np.maximum(X, sigma) + (1 - negative_mask) * Xbar
    return Xbar


def gradient_factors(X, U, V):
    # X [m,n], U [m, r], V[n, r]
    estimate_error = U @ V.T - X
    Ugrad = estimate_error @ V
    Vgrad = estimate_error.T @ U
    return (Ugrad, Vgrad)


def gradient_left_factor(X, U, V):
    # X [m,n], U [m, r], V[n, r]
    estimate_error = U @ V.T - X
    Ugrad = estimate_error @ V
    return Ugrad


class NMF_GRADMUL(NMFBase):
    def __init__(self, params, method_name="GRADMUL"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        print("self.save_dir", self.save_dir)
        print("self.iter_save_dir", self.iter_save_dir)
        # set initial factors
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        # (TODO) rerun times
        self.rerun_times = 1
        # self.dataset_name = "exp"
        self.target_error = 0
        self.max_iter = 300000 * 10 ** (20)
        self.sigma = 0.01  # Ubar, Vbar threshold
        self.delta = 10 ** (-20)  # denominator
        self.tol = 10 ** (-30)  # controls convergence condidtion
        self.target_run_time = None

    def factor_init(self, params):
        if "U" not in params:
            m, n = self.X.shape
            Uinit = abs(np.random.rand(m, self.r))
            Vinit = abs(np.random.rand(n, self.r))
            self.U = Uinit
            self.V = Vinit
        else:
            self.U = params["U"]
            self.V = params["V"]

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
        self.set_params({"target_run_time": target_run_time})

        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = obj > self.target_error
            time_cond = cur_time < self.target_run_time
            return iter_cond and error_cond and time_cond

        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_run_time": target_run_time})
            self.cur_run_id += 1
            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: GRADMUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_tc.npy"
            # save time and error for given iters.
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)

    def run_to_target_error(self, target_error, save_time_error=False):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = obj > target_error
            return iter_cond and error_cond

        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_error": target_error})
            self.cur_run_id += 1
            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: GRADMUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)

    def update_one_factor(self, X, U, V):
        ## Update U : X[m, n], U[m,r], V[n,r]
        Ugrad = gradient_left_factor(X, U, V)
        Ubar = findbar(U, Ugrad, self.sigma)
        UbarVTV = Ubar @ (V.T @ V)
        # option1:
        # U = U - np.multiply(Ugrad, Ubar / (UbarVTV + self.delta))
        # note: this converges much faster than option1
        U = U - np.multiply(Ugrad, Ubar / np.maximum(UbarVTV, self.delta))
        return U

    def one_iter(self, X, U, V, trace_XTX, previous_obj):
        U = self.update_one_factor(X, U, V)
        V = self.update_one_factor(X.T, V, U)

        obj = calculate_obj_NMF(X, U, V, trace_XTX)
        rel = (previous_obj - obj) / previous_obj

        if abs(rel) < self.tol:
            print("Algorithm has converged!")
            return U, V, obj, True
        return U, V, obj, False

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        # default run with tracking of time
        start_t = time.time()
        time_list = []
        error_list = []
        # copy data for further computation
        X = self.X
        U = copy.deepcopy(self.U)
        V = copy.deepcopy(self.V)
        trace_XTX = np.trace(X.T @ X)
        previous_obj = calculate_obj_NMF(X, U, V, trace_XTX)
        n_iter = 0
        continue_cond = True
        exit_cond = False

        while continue_cond and (not exit_cond):
            U, V, obj, exit_cond = self.one_iter(X, U, V, trace_XTX, previous_obj)
            cur_time = time.time() - start_t
            continue_cond = f_continue_cond(n_iter, obj, cur_time)

            exit_cond = (not continue_cond) or exit_cond
            n_iter += 1
            if (n_iter % 50 == 0) or exit_cond:
                print("Saving, checking exit_conition", exit_cond)
                if save_time_error:
                    self.tracker(
                        U,
                        V,
                        self.iter_save_dir,
                        n_iter,
                        {"iter_time": cur_time, "iter_error": obj},
                    )
                else:
                    self.tracker(U, V, self.iter_save_dir, n_iter)
            if verbose and (n_iter % 200 == 0):
                print(f"------GRADMUL comes into {n_iter} iter with loss {obj}")
            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        return U, V, time_list, error_list


def test():
    method_name = "GRADMUL"
    data_dir = os.getcwd()
    print(os.getcwd())
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_exp1.json")

    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print(org_data_mat.shape)
    params = {"X": org_data_mat, "dataset_name": "Verb", "r": 20}
    print(os.getcwd())
    nmf_gradmul = NMF_GRADMUL(method_name=method_name, params=params)
    # nmf_gradmul.run_within_fixed_time(target_run_time=20)
    nmf_gradmul.run_to_target_error(target_error=13.6)


if __name__ == "__main__":
    test()
