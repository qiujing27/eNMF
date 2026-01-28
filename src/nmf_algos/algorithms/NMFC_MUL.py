"""MUL LEE with missing entries. Python version implemented similarly as R package NNLM."""

import os
import time
import copy
import numpy as np
import numpy.linalg as LA
from nmf_algos.utils.linalg_utils import normalize_column_pair, project_error
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.algo_utils import calculate_obj_NMF, HALS_iter_solver
from nmf_algos.NMF_base import NMFBase
from multiprocessing import Pool


TINY_NUM = 1e-16  #


# keep the same notations to be consist with original algorithm.
def mul_ls_update_column(Hj, mu, WtW, inner_rel_tol, inner_max_iter):
    """Update Hj column, mu vector, rel_err"""
    # Hj:r, WtW:r*r,
    rel_err = 1 + inner_rel_tol
    inner_iter = 0
    r = Hj.shape[0]
    # print("Hj", Hj.shape)
    while (inner_iter < inner_max_iter) and (rel_err > inner_rel_tol):
        rel_err = 0
        for k in range(r):
            tmp = Hj.T @ WtW[:, k]
            # Project element into positive orthant.
            tmp = mu[k] / (tmp + TINY_NUM)
            Hj[k] *= tmp
            tmp = 2 * np.abs(tmp - 1) / (tmp + 1)
            rel_err = max(rel_err, tmp)
        inner_iter += 1
        # print("inner iter",inner_iter,  Hj)
    return Hj, rel_err


# # Non parallel version
# def mul_ls_update_factor(A, W, H, known_mask, inner_rel_tol, inner_max_iter, beta):
#     # ||A - WH||, solve H. beta(1)=0, beta(2) = 0, beta refers to beta(0).
#     # A [m, n], W[m, r], H[r,n], known_mask [m. n]
#     r, n = H.shape
#     #print("H shape scd_ls_update_factor", r, n)
#     #print("mask shape", known_mask.shape)
#     #WtW.flat[::r+1]+= TINY_NUM + beta
#     #print("data and mask", A[:10,0], known_mask[:10,0])
#     known_mask_int = known_mask.astype(int)
#     for j in range(n):
#         # Describe which row to be masked.
#         #[m,]
#         row_mask = known_mask[:, j]
#         num_known_rows = np.sum(row_mask)
#         #print("number of known rows", num_known_rows)
#         # if all rows unknow, no need to update
#         if num_known_rows == 0:
#             continue
#         # [g, r]
#         W_known = W[row_mask, :]
#         # [r, r]
#         WtW = W_known.T@W_known
#         #print("wtW shape", WtW.shape)
#         WtW.flat[::r+1]+= beta
#         # [g, n]
#         Aj_known = A[:, j][row_mask]
#         # [r, n]
#         mu = W_known.T@Aj_known
#         H[:, j], rel_err = mul_ls_update_column(H[:, j], mu, WtW, inner_rel_tol, inner_max_iter)
#         print("j:", j, "error:", project_error(A, W, H.T, known_mask_int)[0])
#     return H, rel_err


def mul_ls_update_column_parallel(args):
    Aj, W, Hj, row_mask, beta, inner_rel_tol, inner_max_iter = args
    num_known_rows = np.sum(row_mask)
    # if all rows unknow, no need to update
    if num_known_rows == 0:
        return Hj
    r = Hj.shape[0]
    # [g, r]
    W_known = W[row_mask, :]
    # [r, r]
    WtW = W_known.T @ W_known
    WtW.flat[:: r + 1] += beta
    # [g, n]
    Aj_known = Aj[row_mask]
    # [r, n]
    mu = W_known.T @ Aj_known
    Hj, rel_err = mul_ls_update_column(Hj, mu, WtW, inner_rel_tol, inner_max_iter)
    return Hj


def mul_ls_update_factor(A, W, H, known_mask, inner_rel_tol, inner_max_iter, beta):
    # ||A - WH||, solve H. beta(1)=0, beta(2) = 0, beta refers to beta(0).
    # A [m, n], W[m, r], H[r,n], known_mask [m. n]
    r, n = H.shape
    result_arr = np.zeros_like(H)
    with Pool() as pool:
        # call the same function with different data in parallel
        column_idx = 0
        # for result in pool.imap(scd_ls_update_column_parallel, [{"Aj": A[:,j], "W": W, "Hj": H[:, j],"row_mask":known_mask[:, j],"beta": beta, "inner_rel_tol": inner_rel_tol, "inner_max_iter":inner_max_iter} for j in range(n)]):
        for result in pool.imap(
            mul_ls_update_column_parallel,
            [
                (
                    A[:, j],
                    W,
                    H[:, j],
                    known_mask[:, j],
                    beta,
                    inner_rel_tol,
                    inner_max_iter,
                )
                for j in range(n)
            ],
        ):
            result_arr[:, column_idx] = result
            column_idx += 1
    return result_arr, 0


class NMFC_MUL(NMFBase):
    def __init__(self, params, method_name="MUL"):
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
        self.max_iter = 500
        self.target_run_time = None
        self.beta = 10  # param for SCD
        self.rel_tol = 10 ** (-3)  # controls exit condition
        self.inner_max_iter = 10  # Maximum number of iterations passed to each inner W or H matrix updating loop
        self.inner_rel_tol = 1e-8  # Relative tolerance passed to inner W or H matrix updating loop, = |e2-e1|/avg(e1, e2)

    def factor_init(self, params):
        if "known_mask" not in params:
            raise NotImplementedError(
                "Please provide a known_mask to separate out missing entries in X!"
            )
        self.known_mask = params["known_mask"]
        if "U" not in params:
            np.random.seed(self.cur_run_id)
            m, n = self.X.shape
            self.U = 0.01 * np.random.uniform(size=(m, self.r))
            self.V = 0.01 * np.random.uniform(size=(n, self.r))

    def one_iter(self, A, W, H, previous_err):
        bool_known_mask = self.known_mask.astype(bool)
        H, _ = mul_ls_update_factor(
            A, W, H, bool_known_mask, self.inner_rel_tol, self.inner_max_iter, self.beta
        )
        obj, _ = project_error(A, W, H.T, self.known_mask)
        print("factor one side", obj)
        Wt, _ = mul_ls_update_factor(
            A.T,
            H.T,
            W.T,
            bool_known_mask.T,
            self.inner_rel_tol,
            self.inner_max_iter,
            self.beta,
        )
        W = Wt.T
        obj, _ = project_error(A, W, H.T, self.known_mask)
        print("factor another side", obj)
        current_errr = 0.5 * obj
        rel_err = (
            2 * (previous_err - current_errr) / (previous_err + current_errr + TINY_NUM)
        )
        exit_condition = np.abs(rel_err) <= self.rel_tol
        return W, H, obj, exit_condition

    def basic_run(self, save_time_error=True):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            return iter_cond

        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.cur_run_id += 1
            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: HALS_verb
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
        previous_obj, _ = project_error(self.X, self.U, self.V, self.known_mask)
        print("start obj", previous_obj)
        n_iter = 0
        continue_cond = True
        exit_cond = False

        # Rename variables to be consistent with original algorithm
        W = copy.deepcopy(self.U)
        H = copy.deepcopy(self.V).T
        while continue_cond and (not exit_cond):
            W, H, obj, exit_cond = self.one_iter(self.X, W, H, previous_obj)
            print("n_iter:", n_iter, ": ", obj)
            cur_iter_left_factor, cur_iter_right_factor = W, H.T
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
            previous_obj = obj

        return W, H.T, time_list, error_list


def test():
    method_name = "MUL"
    data_dir = os.getcwd()
    print(os.getcwd())
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_NMFC.json")
    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print("Loaded data matrix with shape:", org_data_mat.shape)
    for r in [5, 10, 15, 20, 25]:
        params = {
            "X": org_data_mat,
            "dataset_name": "Movielens",
            "r": r,
            "rerun_times": 2,
        }
        params["known_mask"] = (org_data_mat > 0).astype(int)
        nmfc_mul = NMFC_MUL(method_name=method_name, params=params)
        nmfc_mul.basic_run()


if __name__ == "__main__":
    test()
