import numpy as np
import time
import os
import shutil

from nmf_algos.utils.utils import  load_data_basedon_proto, load_data_matrix
from nmf_algos import NMF_ENMF, NMF_HALS, NMF_AOADMM, NMF_MUL, NMF_GRADMUL, NMF_ALS

# TODO: ENMF multiple runs. Check verbose option, only print save dir here.
# Run to target error, if not reached with three hour, stop anyway.
project_dir = os.path.join(os.getcwd())
proto_path = os.path.join(project_dir, "Experiments/configs/exact_data_algo_exp.json")
dataset_config = load_data_basedon_proto(proto_path, mode="exactDatasets")
latent_dim = 50
target_run_time = 600
method_name_list = ["HALS", "MUL", "AOADMM", "GRADMUL", "ALS"]
clean_previous_result = True

for dataset_config in dataset_config.exact_dataset:
    for method_name in method_name_list:
        start_t = time.time()
        f_path = os.path.join(
            project_dir, dataset_config.data_dir, dataset_config.data_path
        )
        data_dict = load_data_matrix(f_path)
        org_data_mat = data_dict["X"]
        print(org_data_mat.shape)
        print(dataset_config.data_path)
        params = {
            "X": org_data_mat,
            "dataset_name": dataset_config.name,
            "r": latent_dim,
        }
        instance_name = f"NMF_{method_name}"
        method_instance = globals()[instance_name](
            method_name=method_name, params=params
        )
        print(method_instance.iter_save_dir)
        method_instance.run_within_fixed_time(target_run_time=target_run_time)
        # print(f"Finished Method {method_name} in {time.time()- start_t} seconds")
