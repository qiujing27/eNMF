"""Our refined version of ADMM for NMF problem from Hajinezhad 2016 based on Boyd 2011."""

import os
import time
import copy
import numpy as np
import numpy.linalg as LA
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.algo_utils import calculate_obj_NMF
from nmf_algos.NMF_base import NMFBase


class NMF_ADMM(NMFBase):
    def __init__(self, params, method_name="ADMM"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        print("self.save_dir", self.save_dir)
        print("self.iter_save_dir", self.iter_save_dir)
        # set initial factors
        # self.factor_init(params)

    def reset_status(self, params):
        # prepare for rerun
        self.method_config_init(params)
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        # (TODO) rerun times
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.max_iter = 20000000
        self.mul_tol = 10 ** (-30)
        self.epsilon = 10 ** (-10)
        self.delta_Y = np.inf
        self.delta_D = np.inf
        self.rho = 5
        self.target_run_time = None

    def factor_init(self, params):
        if "U" not in params:
            m, n = self.X.shape
            np.random.seed(self.cur_run_id)
            self.U = np.random.rand(m, self.r)
            self.V = np.random.rand(n, self.r)
            self.Ainit = np.random.rand(m, self.r)
            self.Binit = np.random.rand(n, self.r)
            self.Yinit = np.random.rand(m, self.r)
            self.Dinit = np.random.rand(n, self.r)

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
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
            # model_name: MUL_verb:
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
            # model_name: MUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)

    def update_one_factor(self, X, V, A, rho, Y, D):
        new_U = A - (1 / rho) * Y
        denominator = LA.inv(np.matmul(np.transpose(A), A) + rho * np.eye(A.shape[1]))
        numerator = rho * V + D + np.matmul(np.transpose(X), A)
        new_B = np.matmul(numerator, denominator)
        new_B[new_B < 0] = 0
        return new_U, new_B

    def one_iter(self, X, U, V, A, B, Y, D, trace_XTX):
        ## Update U : X[m, n], U[m,r], V[r,n]
        pre_B = copy.deepcopy(B)
        U, B = self.update_one_factor(X, V, A, self.rho, Y, D)
        V, A = self.update_one_factor(X.T, U, B, self.rho, D, Y)
        update_Y = self.rho * (U - A)
        Y = Y + update_Y
        update_D = self.rho * (V - pre_B)
        D = D + update_D

        delta_Y = LA.norm(update_Y)
        delta_D = LA.norm(update_D)
        exit_condition = (delta_Y <= self.epsilon) or (delta_D <= self.epsilon)
        obj = calculate_obj_NMF(X, U, V, trace_XTX)
        return U, V, A, B, Y, D, obj, exit_condition

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        # default run with tracking of time
        start_t = time.time()
        time_list = []
        error_list = []
        # copy data for further computation
        X = self.X
        U = copy.deepcopy(self.U)
        V = copy.deepcopy(self.V)
        A = self.Ainit
        B = self.Binit
        Y = self.Yinit
        D = self.Dinit
        trace_XTX = np.trace(X.T @ X)
        previous_obj = calculate_obj_NMF(X, U, V, trace_XTX)
        n_iter = 0
        continue_cond = True
        exit_cond = False

        while continue_cond and (not exit_cond):
            U, V, A, B, Y, D, obj, exit_cond = self.one_iter(
                X, U, V, A, B, Y, D, trace_XTX
            )
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
                print(f"------ADMM comes into {n_iter} iter with loss {obj}")
            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        return U, V, time_list, error_list

    def basic_run(self, save_time_error=True):
        # only controlled by number of iterations
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            return iter_cond

        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"max_iter": 1000})
            self.cur_run_id += 1

            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: MUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_default.npy"
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)


def test():
    data_dir = os.getcwd()
    print(os.getcwd())
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_exp1.json")
    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print(org_data_mat.shape)
    # print(real_dataset.name)
    # params = {"X": org_data_mat, "r": 20, "save_dir": "", "iter_save_dir": }
    latent_dims = [10, 20, 40, 80, 100]
    for latent_dim in latent_dims:
        params = {"X": org_data_mat, "dataset_name": "Verb", "r": latent_dim}

        nmf_admm = NMF_ADMM(params=params)
        # Run algorithm once with max_iter=1000.
        nmf_admm.basic_run()
    # nmf_admm.run_within_fixed_time(target_run_time=20)
    # nmf_admm.run_to_target_error(target_error=13.6)


if __name__ == "__main__":
    test()
