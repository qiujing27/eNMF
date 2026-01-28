"""ADM(Alternating Direction Method, Xu, 2012), a baseline algorithm for data matrix with missing entries."""

import os
import time
import copy
import numpy as np
import numpy.linalg as LA
from nmf_algos.utils.linalg_utils import normalize_column_pair, project_error
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.algo_utils import calculate_obj_NMF, HALS_iter_solver
from nmf_algos.NMF_base import NMFBase


class NMFC_ADM(NMFBase):
    def __init__(self, params, method_name="ADM"):
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
        self.max_iter = 1000
        self.target_run_time = None
        self.tol = 10 ** (-3)

    def factor_init(self, params):
        if "known_mask" not in params:
            raise NotImplementedError(
                "Please provide a known_mask to separate out missing entries in X!"
            )
        self.known_mask = params["known_mask"]
        if "U" not in params:
            np.random.seed(self.cur_run_id)
            m, n = self.X.shape
            Uinit = abs(np.random.rand(m, self.r))
            Vinit = abs(np.random.rand(n, self.r))
            self.U = Uinit
            self.V = Vinit
        # ADM params
        self.gamma = 1.618
        self.alpha = (50 * max(m, n)) / self.r
        self.beta = (self.alpha * n) / m
        # To stablize the inverse operation.
        self.lamda = 10

    def one_iter(self, A, X, Y, Z, U, V, Delta, Kappa, trace_ATA):
        # To make the original algorithm easy to follow, use same annotations as in the paper.
        # An Alternating Direction Algorithm for Matrix Completion with Nonnegative Factors.
        # A: data with unknow entries.
        f_prev_num = LA.norm(np.multiply(self.known_mask, np.matmul(X, Y) - A))
        f_prev_den = LA.norm(A)
        f_prev = f_prev_num / f_prev_den
        # Updating X
        X_t1 = np.matmul(Z, Y.T) + self.alpha * U - Delta
        X_t2 = np.matmul(Y, Y.T) + (self.alpha + self.lamda) * np.eye(Y.shape[0])
        X = np.matmul(X_t1, LA.inv(X_t2))
        # Updating Y
        Y_t1 = np.matmul(X.T, X) + (self.beta + self.lamda) * np.eye(X.shape[1])
        Y_t2 = np.matmul(X.T, Z) + self.beta * V - Kappa
        Y = np.matmul(LA.inv(Y_t1), Y_t2)
        # Updating Z
        Z_t1 = np.matmul(X, Y)
        Z_t2 = A - np.matmul(X, Y)
        Z_t2_proj = np.multiply(self.known_mask, Z_t2)
        Z = Z_t1 + Z_t2_proj
        # Updating U
        U = X + Delta / self.alpha
        U[U < 0] = 0
        # Updating V
        V = Y + Kappa / self.beta
        V[V < 0] = 0
        # Updating Delta
        Delta = Delta + self.gamma * self.alpha * (X - U)
        # Updating Kappa
        Kappa = Kappa + self.gamma * self.beta * (Y - V)
        f_cur_num = LA.norm(np.multiply(self.known_mask, np.matmul(X, Y) - A))
        f_cur_den = LA.norm(A)
        f_cur = f_cur_num / f_cur_den

        error1_num = abs(f_cur - f_prev)
        error1_den = max(1, abs(f_prev))
        error1 = error1_num / error1_den
        error2 = f_prev
        # Decide exit condition:
        exit_condition = (error1 <= self.tol) and (error2 <= self.tol)
        obj, _ = project_error(A, X, Y.T, self.known_mask)
        return X, Y, Z, U, V, Delta, Kappa, obj, exit_condition

    def basic_run(self, save_time_error=True):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            return iter_cond

        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.cur_run_id += 1
            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: HALS_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_default.npy"
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        # default run with tracking of time
        start_t = time.time()
        time_list = []
        error_list = []
        # copy data for further computation
        trace_XTX = np.trace(self.X.T @ self.X)
        previous_obj = calculate_obj_NMF(self.X, self.U, self.V, trace_XTX)
        print("start obj", previous_obj)
        n_iter = 0
        continue_cond = True
        exit_cond = False

        # Rename variables to be consistent with original algorithm
        A = self.X
        X = copy.deepcopy(self.U)
        Y = copy.deepcopy(self.V.T)
        Z = copy.deepcopy(self.X)
        m, n = self.X.shape
        U = np.zeros((m, self.r))
        V = np.zeros((self.r, n))
        Delta = np.zeros((m, self.r))
        Kappa = np.zeros((self.r, n))
        while continue_cond and (not exit_cond):
            X, Y, Z, U, V, Delta, Kappa, obj, exit_cond = self.one_iter(
                A, X, Y, Z, U, V, Delta, Kappa, trace_XTX
            )
            cur_iter_left_factor, cur_iter_right_factor = X, Y.T
            cur_time = time.time() - start_t
            continue_cond = f_continue_cond(n_iter, obj, cur_time)
            exit_cond = (not continue_cond) or exit_cond
            n_iter += 1
            if (n_iter % 50 == 0) or exit_cond:
                print("Saving, checking exit_conition", exit_cond)
                if save_time_error:
                    self.tracker(
                        cur_iter_left_factor,
                        cur_iter_right_factor,
                        self.iter_save_dir,
                        n_iter,
                        {"iter_time": cur_time, "iter_error": obj},
                    )
                else:
                    self.tracker(
                        cur_iter_left_factor,
                        cur_iter_right_factor,
                        self.iter_save_dir,
                        n_iter,
                    )
            if verbose and (n_iter % 200 == 0):
                print(f"------ADM comes into {n_iter} iter with loss {obj}")
            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        return X, Y.T, time_list, error_list


def test():
    method_name = "ADM"

    data_dir = os.getcwd()
    print(os.getcwd())
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_NMFC.json")
    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print("Loaded data matrix with shape:", org_data_mat.shape)
    # Save dir would be: ENMF/Results/{dataset_name}/{method_name}/latent_dim_{r}/{rerun_times}/Iters.
    for r in [5, 10, 15, 20, 25]:
        params = {
            "X": org_data_mat,
            "dataset_name": "Movielens",
            "r": r,
            "rerun_times": 2,
        }
        params["known_mask"] = (org_data_mat > 0).astype(int)

        nmfc_adm = NMFC_ADM(method_name=method_name, params=params)
        nmfc_adm.basic_run()


if __name__ == "__main__":
    test()
