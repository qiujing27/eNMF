"""eNMC, an extension of eNMF algorithm for data matrix with missing entries."""

import os
import time
import copy
import numpy as np
import numpy.linalg as LA
from nmf_algos.utils.linalg_utils import  project_error
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.ENMF_utils import  move_to_positive_orthant
from .NMF_ENMF import NMF_ENMF


# softImpute_ALS [Hastie, 2015]
# Computes local minimum of the unconstrained problem. ||M_E(X-UV^T)||
def obj_softimpute_UV(X, U, V, known_mask, lamda):
    term1 = 0.5 * np.square(LA.norm(np.multiply(known_mask, X - np.matmul(U, V.T))))
    term2 = lamda * 0.5 * (np.square(LA.norm(U)) + np.square(LA.norm(V)))
    return term1 + term2, term1, term2


def softImpute_ALS(X, U, V, known_mask):
    lamda = 10
    maxiter = 1000
    tol = 10 ** (-3)  # 10**(-1)
    errorU = np.inf
    errorV = np.inf
    n_iter = 0
    while ((errorU > tol) or (errorV > tol)) and (n_iter < maxiter):
        UVt = np.matmul(U, V.T)
        projected_1 = np.multiply(known_mask, X - UVt)
        X_star1 = projected_1 + UVt
        VtV = np.matmul(V.T, V)
        inv_VtV = LA.inv(VtV + lamda * np.eye(V.shape[1]))
        pre_U = copy.deepcopy(U)
        U = np.matmul(np.matmul(X_star1, V), inv_VtV)
        UVt = np.matmul(U, V.T)
        projected_2 = np.multiply(known_mask, X - UVt)
        X_star2 = projected_2 + UVt
        UtU = np.matmul(U.T, U)
        inv_UtU = LA.inv(UtU + lamda * np.eye(U.shape[1]))
        pre_V = copy.deepcopy(V)
        V = np.matmul(np.matmul(np.transpose(X_star2), U), inv_UtU)
        errorU = LA.norm(U - pre_U)
        errorV = LA.norm(V - pre_V)
        n_iter += 1
        if n_iter % 100 == 0:
            print(f"iter {n_iter}", project_error(X, U, V, known_mask))
            # obj_val, t1, t2 = obj_softimpute_UV(X, U, V, known_mask, lamda)
            # print("iter {}: objective value {}, re {}, norm(U) + norm(V) {},".format(n_iter, obj_val, t1, t2))
            # print("errorU {}, errorV {} ".format(errorU, errorV))
    return U, V


class NMFC_ENMF(NMF_ENMF):
    def __init__(self, params, method_name="ENMFC"):
        # Only inherit basic setting instead of intializing with all ENMF configs.
        super(NMF_ENMF, self).__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        print("self.save_dir", self.save_dir)
        print("self.iter_save_dir", self.iter_save_dir)
        # set initial factors
        # self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        # (TODO) rerun times
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.eps = 10 ** (-16)
        # self.eps = 1e-10 # maximum bound
        # self.max_iter = 300000 * 10 ** (20)
        self.target_run_time = None
        self.enmfc_config_init()
        self.intermediate_result_dict = {}

    def enmfc_config_init(self):
        self.admm_config = {
            "rho": 5,
            "epsilon": 10 ** (-4),
            "max_iter": 4000,
            "tau_inc": 1.1,
            "tau_dec": 1.1,
            "mu": 2,
            "rho_mode": 0,
        }
        self.ascent_config = {"tol_asc": 0.2, "inner_iter_asc": 2, "num_steps": 1000}
        # self.descent_config = {"hals_rounds": 10**2}
        combined_config_dict = {}
        combined_config_dict.update(self.admm_config)
        combined_config_dict.update(self.ascent_config)
        # combined_config_dict.update(self.descent_config)
        for key, value in combined_config_dict.items():
            # setattr(NMF_ENMC, key, value)
            setattr(NMFC_ENMF, key, value)

    def factor_init(self, params):
        if "known_mask" not in params:
            raise NotImplementedError(
                "Please provide a known_mask to separate out missing entries in X!"
            )
        self.known_mask = params["known_mask"]
        if "softimpute_U" not in params:
            # Use softImpute_ALS to initialize U and V factors.
            m, n = self.X.shape
            np.random.seed(self.cur_run_id)
            Uinit = np.random.rand(m, self.r)
            Vinit = np.random.rand(n, self.r)
            start_t = time.time()
            # Pseudo svd factors from softImpute_ALS
            self.U_svd, self.V_svd = softImpute_ALS(
                self.X, Uinit, Vinit, self.known_mask
            )
            self.t_svd = time.time() - start_t
            print("Step1: factors initialization via softimpute")

        self.trace_XTX = np.trace(self.X.T @ self.X)
        self.svd_error, _ = project_error(
            self.X, self.U_svd, self.V_svd, self.known_mask
        )

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
            self.known_mask,
        )
        self.t_mp = time.time() - start_t
        self.hitmp_error, _ = project_error(
            self.X, self.U_mp, self.V_mp, self.known_mask
        )
        print("Step3: t_mp ", self.t_mp)

    def store_intermedia_results(self):
        # svd_dict = {"U_eig": self.U_svd, "V_eig": self.V_svd, "t_svd": self.t_svd, "svd_error": self.svd_error}
        softimpute_result = {
            "U_softimpute": self.U_svd,
            "V_softimpute": self.V_svd,
            "softimpute_error": self.svd_error,
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
            "enmf_error": self.enmf_error,
            "total_time": self.total_runtime,
        }
        self.intermediate_result_dict.update(softimpute_result)
        self.intermediate_result_dict.update(rotated_dict)
        self.intermediate_result_dict.update(mp_dict)
        # self.intermediate_result_dict.update(descent_dict)

    def core_run(self):
        self.move_to_PO()
        self.U, self.V = self.U_mp, self.V_mp
        # self.t_descent = 0
        self.total_runtime = self.t_mp + self.t_svd + self.t_rotate
        file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_default.npy"

        self.enmf_error, re_error_relative = project_error(
            self.X, self.U_mp, self.V_mp, self.known_mask
        )
        # Move results to intermediate_result_dict.
        self.store_intermedia_results()
        # save time and error, and intermedia results for ENMF
        self.save_factors(file_name, self.intermediate_result_dict)

    def basic_run(self):
        # ENMC will only run PBCD without descending.
        # don't have control over time, so don't provie save_time_error list option.
        # it is controlled by max iteration
        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.cur_run_id += 1
            self.core_run()


def test():
    method_name = "ENMFC"
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
            "rerun_times": 1,
        }
        params["known_mask"] = (org_data_mat > 0).astype(int)
        nmfc_enmf = NMFC_ENMF(method_name=method_name, params=params)
        nmfc_enmf.basic_run()


if __name__ == "__main__":
    test()
