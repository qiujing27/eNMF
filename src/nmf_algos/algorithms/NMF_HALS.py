import os
import time
import copy
import numpy as np
import numpy.linalg as LA
from nmf_algos.utils.linalg_utils import normalize_column_pair
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.algo_utils import calculate_obj_NMF, HALS_iter_solver
from nmf_algos.NMF_base import NMFBase


class NMF_HALS(NMFBase):
    def __init__(self, params, method_name="HALS"):
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
        self.dataset_name = "exp"
        self.target_error = 0
        self.eps = 10 ** (-16)
        # self.eps = 1e-10 # maximum bound
        self.max_iter = 300000 * 10 ** (20)
        self.target_run_time = None

    def factor_init(self, params):
        if "U" not in params:
            m, n = self.X.shape
            np.random.seed(self.cur_run_id)
            Uinit = abs(np.random.rand(m, self.r))
            Vinit = abs(np.random.rand(n, self.r))
            Uinit, Vinit, _ = normalize_column_pair(Uinit, Vinit)
            self.U = Uinit
            self.V = Vinit
            print("initilized factors")
        else:
            self.U = params["U"]
            self.V = params["V"]

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
            # model_name: HALS_verb:
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
            # model_name: HALS_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)

    def one_iter(self, X, U, V, trace_XTX, previous_obj):
        U, V = HALS_iter_solver(X, U, V, self.r, self.eps)
        obj = calculate_obj_NMF(X, U, V, trace_XTX)
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
        print("start obj", previous_obj)
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
                print(f"------HALS comes into {n_iter} iter with loss {obj}")
            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        return U, V, time_list, error_list


def test():
    method_name = "HALS"
    data_dir = os.getcwd()
    print(os.getcwd())
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_exp1.json")
    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print(org_data_mat.shape)

    latent_dims = [10, 20, 40, 80, 100]
    target_time = [360, 360, 360, 360, 360]
    for latent_dim, target_t in zip(latent_dims, target_time):
        params = {
            "X": org_data_mat,
            "dataset_name": "Verb",
            "r": latent_dim,
            "rerun_times": 1,
        }
        print(os.getcwd())
        nmf_hals = NMF_HALS(method_name=method_name, params=params)
        nmf_hals.run_within_fixed_time(target_run_time=target_t)
        # nmf_hals.run_to_target_error(target_error=target_error)


if __name__ == "__main__":
    test()
