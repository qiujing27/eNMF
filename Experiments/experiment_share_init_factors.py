
import numpy as np
import time
import os
import pickle
from nmf_algos.utils.utils import load_data_matrix
from nmf_algos import NMF_ENMF, NMF_HALS, NMF_AOADMM, NMF_MUL, NMF_GRADMUL, NMF_ALS
from initialization_algos.init_algo import get_init_factors

# UV factors will be saved into Results/Verb/method_name/latent_dim_{latent_dim}/{init_method}
# Initialized factors will be saved into Results/Verb/Init/r_{latent_dim}/{init_method}
method_name_list = ["HALS", "MUL","AOADMM", "GRADMUL", "ALS"]
latent_dim_list = [20] #[10, 20, 40, 80, 100]
project_dir = os.getcwd()
f_path = os.path.join(project_dir, "Dataset/verb/right_matrix.npy")
org_data_mat = load_data_matrix(f_path)
print("Loaded data with shape: ", org_data_mat.shape)
init_factor_save_dir = os.path.join(project_dir, "Results/Verb/Init")

os.makedirs(init_factor_save_dir, exist_ok=True) 
init_method_list = ["pso", "de", "fss", "random"]
#init_method_list = ["random", "nndsvdar", "kmeans","nica"]
for init_method in init_method_list:
    for latent_dim in latent_dim_list:
        U, V = get_init_factors(org_data_mat, latent_dim, init_method=init_method)
        print("U.shape",U.shape, "V.shape", V.shape)
        print(f"Finished initialization with {init_method}")
        
        data_dict = {"U": U, "V": V,  "init_method": init_method, "r": latent_dim}
        with open(os.path.join(init_factor_save_dir, f"r_{latent_dim}_{init_method}.pkl"), "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        for method_name in method_name_list:
            result_save_dir = os.path.join(project_dir, f"Results/Verb/{method_name}/latent_dim_{latent_dim}/init_{init_method}")
            os.makedirs(result_save_dir, exist_ok=True)
            start_t = time.time()
            params = {"X": org_data_mat, "U":U, "V": V.T, "dataset_name": "Verb", "r": 20,  "save_dir": result_save_dir}
            instance_name = f"NMF_{method_name}"
            method_instance = globals()[instance_name](method_name=method_name, params=params)
            method_instance.run_within_fixed_time(target_run_time=20)
            print(f"Finished Method {method_name} in {time.time()- start_t} seconds")

    

