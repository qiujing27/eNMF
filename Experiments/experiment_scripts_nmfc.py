import numpy as np
import time
import os

from nmf_algos.utils.utils import load_data_basedon_proto, fetch_factors_from_result_path
from nmf_algos import NMFC_ADM, NMFC_ENMF, NMFC_MUL, NMFC_SCD
#TODO: ENMF multiple runs. Check verbose option, only print save dir here.
# Run to target error, if not reached with three hour, stop anyway.  
project_dir = os.path.join(os.getcwd())
proto_path = os.path.join(project_dir, "Experiments/configs/real_data_algo_NMFC.json")
dataset_configs = load_data_basedon_proto(proto_path, mode="synDatasets").real_dataset
# Option1: use a given latent_dim
# latent_dim = 500
# Option2: use latent dim parsed from config

for dataset_config in dataset_configs:
    print(dataset_config)
    for method_config in dataset_config.method_config:
        method_name = method_config.method_name
        print(method_name)
        start_t = time.time()
        f_path = os.path.join(project_dir, dataset_config.data_dir, dataset_config.data_path)
        print(f_path)
        latent_dim = method_config.latent_dim
        org_data_mat = fetch_factors_from_result_path(f_path)
        print(org_data_mat.shape)
        params = {"X": org_data_mat, "dataset_name": dataset_config.name, "r": latent_dim}
        params["known_mask"] = (org_data_mat > 0).astype(int)
        instance_name = f"NMFC_{method_name}"
        method_instance = globals()[instance_name](method_name=method_name, params=params)
        print("iter_save_dir", method_instance.iter_save_dir)
        method_instance.basic_run()
        print(f"Finished Method {method_name} in {time.time()- start_t} seconds")