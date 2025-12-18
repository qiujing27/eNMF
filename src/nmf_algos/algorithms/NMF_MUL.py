import os
import time
import copy
import numpy as np
import numpy.linalg as LA
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.algo_utils import calculate_obj_NMF
from nmf_algos.NMF_base import NMFBase

#(TODO) decide 1. normalize UV for each step or not 2. denominator bound by maximum or not
def normalizeUV_noNorm(U, V):
    V_colSum = np.maximum(np.sum(V, axis=0), 1e-10)
    Q = np.diag(V_colSum, 0)
    Qinv = np.diag((1.0/V_colSum), 0)
    U = U @ Q # n-by-K
    V = V @ Qinv # d-by-K
    return (U, V, Q)

class NMF_MUL(NMFBase):
    def __init__(self, params, method_name="MUL"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        print("self.save_dir", self.save_dir)
        print("self.iter_save_dir", self.iter_save_dir)
        # set initial factors
        self.factor_init(params)
        
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
        self.max_iter = 300000 * 10 ** (20)
        self.mul_tol = 10**(-30)
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
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = obj > self.target_error
            time_cond = cur_time < self.target_run_time
            return iter_cond and error_cond and time_cond
        
        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_run_time": target_run_time})
            self.cur_run_id +=1
        
            U, V, time_list, error_list  = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: MUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_tc.npy"
            # save time and error for given iters.
            if save_time_error:
                self.save_factors(file_name, {"iter_time": time_list, "iter_error": error_list})
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
            self.cur_run_id +=1
            
            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: MUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            if save_time_error:
                self.save_factors(file_name, {"iter_time": time_list, "iter_error": error_list})
            else:
                self.save_factors(file_name)

    def update_one_factor(self, X, U, V):
        ## Update U : X[m, n], U[m,r], V[n,r]
        XTU = X.T @ U  # n-by-r
        UTU = U.T @ U  # r-by-r
        VUTU = V @ UTU  # n-by-r
        V = np.multiply(V, XTU / np.maximum(VUTU, 1e-10))
        # Option2 
        # V = np.multiply(V, XTU / (VUTU  + lambda_v * V +  1e-10))
        return V
        
        
    def one_iter(self, X, U, V, trace_XTX, previous_obj):
        ## Update U : X[m, n], U[m,r], V[r,n]
        V = self.update_one_factor(X, U, V)
        U = self.update_one_factor(X.T, V, U)
        U, V, Q = normalizeUV_noNorm(U, V)
        obj = calculate_obj_NMF(X, U, V, trace_XTX)
        rel = (previous_obj - obj) / previous_obj
        if abs(rel) < self.mul_tol:
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
        trace_XTX = np.trace(X.T@X)
        previous_obj = calculate_obj_NMF(X, U, V, trace_XTX)
        n_iter = 0
        continue_cond = True
        exit_cond = False
       
        while continue_cond and (not exit_cond):
            U, V, obj, exit_cond = self.one_iter(
                X, U, V, trace_XTX, previous_obj)
            cur_time = time.time() - start_t
            continue_cond = f_continue_cond(n_iter, obj, cur_time)
            
            exit_cond = (not continue_cond) or exit_cond
            n_iter += 1
            if (n_iter % 50 == 0) or exit_cond:
                print("Saving, checking exit_conition", exit_cond)
                if save_time_error:
                    self.tracker(U, V, self.iter_save_dir, n_iter, {"iter_time": cur_time, "iter_error":obj})
                else:
                    self.tracker(U, V, self.iter_save_dir, n_iter)
            if verbose and (n_iter % 200 == 0):
                print(f"------MUL comes into {n_iter} iter with loss {obj}")
            if  save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        return  U, V, time_list, error_list


def test():
    method_name = "MUL"
    data_dir = os.getcwd()
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_exp1.json")
    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print("Loaded data matrix with shape:", org_data_mat.shape)

    params = {"X": org_data_mat, "dataset_name": "Verb", "r": 20}
    print(os.getcwd())
    nmf_mul = NMF_MUL(method_name=method_name, params=params)
    #nmf_mul.run_within_fixed_time(target_run_time=20)
    nmf_mul.run_to_target_error(target_error=13.6)

if __name__ == "__main__":
    test()
#1.15 Check bug in This algorithm. Test running time of calculate_obj_NMF and get_l2_error. 
# Note: There are two versions of MUL: MUL, accMUL. Two versions of implementation: 1. Ehsan. 2 NMF package
#1.17 (TODO) check V factor shape, to be consistent wtih ALS. 
