import numpy as np
import time
import os

from nmf_algos.utils.utils import load_data_basedon_proto, fetch_factors_from_result_path
from nmf_algos import NMF_ENMF, NMF_HALS, NMF_AOADMM, NMF_MUL, NMF_GRADMUL, NMF_ALS
#TODO: ENMF multiple runs. Check verbose option, only print save dir here.
# Run to target error, if not reached with three hour, stop anyway.  
project_dir = os.path.join(os.getcwd())
proto_path = os.path.join(project_dir, "Experiments/configs/syn_data_algo_exp.json")
dataset_configs = load_data_basedon_proto(proto_path, mode="synDatasets").real_dataset
# Option1: use a given latent_dim
# latent_dim = 500
# Option2: use latent dim parsed from config
target_run_time = 1000
method_name_list = ["HALS", "ADMM","AOADMM"]
clean_previous_result = True

for dataset_config in dataset_configs:
    for method_config in dataset_config.method_config:
        method_name = method_config.method_name
        start_t = time.time()
        f_path = os.path.join(project_dir, dataset_config.data_dir, dataset_config.data_path)
        print( dataset_config.data_path)
        for latent_dim in dataset_config.latent_dim:
            org_data_mat = fetch_factors_from_result_path(f_path)
            print(org_data_mat.shape)
            params = {"X": org_data_mat, "dataset_name": dataset_config.name, "r": latent_dim}
            instance_name = f"NMF_{method_name}"
            method_instance = globals()[instance_name](method_name=method_name, params=params)
            print(method_instance.iter_save_dir)
            method_instance.run_within_fixed_time(target_run_time=target_run_time)
            print(f"Finished Method {method_name} in {time.time()- start_t} seconds")
        
    

