import os
import time
import numpy as np
from numpy import linalg as LA
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.ENMF_utils import (
    gen_svd_sol,
    admm_rotation,
    move_to_positive_orthant,
    HALS_pos
)
from nmf_algos.utils.algo_utils import calculate_obj_NMF

from nmf_algos.NMF_base import NMFBase


class NMF_ENMF(NMFBase):
    def __init__(self, params, method_name="ENMF"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        print("self.save_dir", self.save_dir)
        print("self.iter_save_dir", self.iter_save_dir)
        # set initial factors: moved to basic run
        # self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        # (TODO) rerun times
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.target_run_time = 3600  # run for one hour #None
        # (TODO) refine description: ADMM
        self.mu = 2
        self.rho_mode = 0
        self.normalize_data = False
        self.scale = 1.0
        self.enmf_config_init()
        # Stores intermediate step results special for eNMF algorithm.
        self.intermediate_result_dict = {}

    def data_normalization(self):
        gap = (np.max(self.X) - np.min(self.X)) * 1.0
        self.scaled_X = self.X / gap
        self.scale = gap
        print(
            f"Data min {np.min(self.X)} max {np.max(self.X)}. Scale data by {self.scale}"
        )

    def enmf_config_init(self):
        self.admm_config = {
            "rho": 5,
            "epsilon": 10 ** (-4),
            "max_iter": 4000,
            "tau_inc": 1.1,
            "tau_dec": 1.1,
        }
        self.ascent_config = {"tol_asc": 0.2, "inner_iter_asc": 2, "num_steps": 100}
        self.descent_config = {"hals_rounds": 10**2}
        self.combined_config_dict = {}
        self.combined_config_dict.update(self.admm_config)
        self.combined_config_dict.update(self.ascent_config)
        self.combined_config_dict.update(self.descent_config)
        for key, value in self.combined_config_dict.items():
            setattr(NMF_ENMF, key, value)

    def factor_init(self, params):
        # Load intialized factors if there are.
        if self.normalize_data:
            self.data_normalization()
            self.X = self.scaled_X
        else:
            self.scaled_X = self.X

        if ("U_eig" in params) and ("V_eig" in params):
            self.U_svd = params["U_eig"]
            self.V_svd = params["V_eig"]
            # Ignore the time running for SVD if not provided
            self.t_svd = params["t_svd"] if "t_svd" in params else 0
            print("Step1: Loaded SVD solution")
        else:
            self.get_svd()
            print("t_svd", self.t_svd)
            print("Step1: Generated SVD initilizated factors")
        # self.svd_error = calculate_obj_NMF(self.X, self.U_svd, self.V_svd, self.trace_XTX)
        # Use LA.norm instead due to the NAN rounding error when svd_error close to 0
        self.svd_error = LA.norm(self.X - self.U_svd @ self.V_svd.T)

        if ("U_rotate" in params) and ("V_rotate" in params):
            # extend to case where RCR denote
            self.U_rotation = params["U_rotate"]
            self.V_rotation = params["V_rotate"]
            self.t_rotate = params["t_rotate"] if "t_rotate" in params else 0
            print("Step2: Loaded rotation solution")
        else:
            self.get_rotation()
            print("Step2: Generated rotated factors.")
            print("t_rotation", self.t_rotate)

    def store_intermedia_results(self):
        svd_dict = {
            "U_eig": self.U_svd,
            "V_eig": self.V_svd,
            "t_svd": self.t_svd,
            "svd_error": self.svd_error,
        }
        rotated_dict = {
            "U_rotate": self.U_rotation,
            "V_rotate": self.V_rotation,
            "t_rotate": self.t_rotate,
            "distance_po": self.dist_po,
        }
        mp_dict = {
            "U_mp": self.U_mp,
            "V_mp": self.V_mp,
            "t_mp": self.t_mp,
            "hitmp_error": self.hitmp_error,
        }
        descent_dict = {
            "U_nmf": self.U_nmf,
            "V_nmf": self.V_nmf,
            "t_nmf": self.t_descent,
            "enmf_error": self.enmf_error,
            "total_time": self.total_runtime,
            "data_scale": self.scale,
        }

        self.intermediate_result_dict.update(svd_dict)
        self.intermediate_result_dict.update(rotated_dict)
        self.intermediate_result_dict.update(mp_dict)
        self.intermediate_result_dict.update(descent_dict)

    def get_svd(self):
        """
        Step 1:  Obtaining the SVD solution as initial factors.
        """
        start_t = time.time()
        U_svd, V_svd = gen_svd_sol(self.X, self.r)
        self.t_svd = time.time() - start_t
        self.U_svd = U_svd
        self.V_svd = V_svd

    def get_rotation(self):
        """
        Step 2: Computing the rotated SVD solution closest to the positive orthant.
        """
        W = np.vstack((self.U_svd, self.V_svd))
        start_t = time.time()
        print(
            self.rho,
            self.epsilon,
            self.max_iter,
            self.tau_inc,
            self.tau_dec,
            self.mu,
            self.rho_mode,
        )
        res_R, obj_f1 = admm_rotation(
            W,
            self.rho,
            self.epsilon,
            self.max_iter,
            self.tau_inc,
            self.tau_dec,
            self.mu,
            self.rho_mode,
        )
        self.t_rotate = time.time() - start_t

        UR_star = np.matmul(self.U_svd, res_R)
        VR_star = np.matmul(self.V_svd, res_R)

        self.U_rotation = UR_star
        self.V_rotation = VR_star
        self.dist_po = obj_f1  ## Added this line
        print("self.dist_po", self.dist_po)

    def move_to_PO(self):
        """
        Step 3: Attaining feasibility of the rotated factors using PBCD.
        """
        start_t = time.time()
        self.U_mp, self.V_mp = move_to_positive_orthant(
            self.X,
            self.U_rotation,
            self.V_rotation,
            self.tol_asc,
            self.inner_iter_asc,
            self.num_steps,
            self.dist_po,
        )
        self.t_mp = time.time() - start_t
        self.hitmp_error = calculate_obj_NMF(
            self.X, self.U_mp, self.V_mp, self.trace_XTX
        )
        print("Step3: t_mp ", self.t_mp)

    def descend_to_enmf(self):
        """
        Step 4: Descending to the eNMF factors using HALS.
        """
        start_t = time.time()
        ### compare hit bound error with svd solution: if close enough, skip HALS

        if np.abs(self.hitmp_error - self.svd_error) > 10 ** (-4):
            hals_target_run_time = (
                self.target_run_time - self.t_svd - self.t_mp - self.t_rotate
            )
            self.U_nmf, self.V_nmf = HALS_pos(
                self.X,
                self.trace_XTX,
                self.U_mp,
                self.V_mp,
                self.r,
                self.hals_rounds,
                hals_target_run_time,
                self.target_error,
            )
        else:
            self.U_nmf, self.V_nmf = self.U_mp, self.V_mp
        self.t_descent = time.time() - start_t
        self.total_runtime = self.t_descent + self.t_mp + self.t_svd + self.t_rotate
        self.enmf_error = calculate_obj_NMF(
            self.X, self.U_nmf, self.V_nmf, self.trace_XTX
        )

    def core_run(self):
        self.trace_XTX = np.trace(self.X.T @ self.X)
        self.move_to_PO()
        self.descend_to_enmf()
        self.U = self.U_nmf
        self.V = self.V_nmf
        if self.normalize_data:
            self.rescale_result()

    def rescale_result(self):
        self.X = self.scaled_X * self.scale
        self.U = self.U * self.scale
        self.U_nmf = self.U_nmf * self.scale
        self.trace_XTX = np.trace(self.X.T @ self.X)
        self.enmf_error = calculate_obj_NMF(
            self.X, self.U_nmf, self.V_nmf, self.trace_XTX
        )

    def basic_run(self):
        # Step1 and 2 are done in self.factor_init()
        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.cur_run_id += 1
            self.core_run()
            file_name = f"{self.method_name}_{self.dataset_name}_r_{self.r}_default.npy"
            # Move results to intermediate_result_dict.
            self.store_intermedia_results()
            # save time and error, and intermedia results for ENMF
            self.save_factors(file_name, self.intermediate_result_dict)

    def run_to_target_error(self, target_error, save_time_error=False):
        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_error": target_error, "hals_rounds": 10**10})
            self.cur_run_id += 1

            self.core_run()
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            self.store_intermedia_results()
            # save time and error, and intermedia results for ENMF
            self.save_factors(file_name, self.intermediate_result_dict)

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_run_time": target_run_time, "hals_rounds": 10**10})
            self.cur_run_id += 1

            self.core_run()
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_tc.npy"
            self.store_intermedia_results()
            # save time and error, and intermedia results for ENMF
            self.save_factors(file_name, self.intermediate_result_dict)


def test():
    data_dir = os.getcwd()
    print(os.getcwd())
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_exp1.json")
    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print("Loaded data matrix with shape:", org_data_mat.shape)
    latent_dims = [10, 20, 40, 80, 100]
    for latent_dim in latent_dims:
        params = {"X": org_data_mat, "dataset_name": "Verb", "r": latent_dim}
        print(os.getcwd())
        nmf_enmf = NMF_ENMF(params=params)
        nmf_enmf.basic_run()
        print(nmf_enmf.intermediate_result_dict)


if __name__ == "__main__":
    test()
