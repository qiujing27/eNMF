import numpy as np
import time
import os
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos import NMF_ENMF, NMF_HALS, NMF_AOADMM, NMF_MUL, NMF_GRADMUL, NMF_ALS

# Run to target error, if not reached with three hour, stop anyway.
method_name_list = ["HALS", "MUL", "AOADMM", "GRADMUL", "ALS"]
latent_dim_list = [5, 10, 15, 20, 25]
project_dir = os.path.join(os.getcwd())
f_path = os.path.join(project_dir, "Dataset/face_id_4.npy")
org_data_mat = load_data_matrix(f_path) + 255.0
print("Loaded data with shape: ", org_data_mat.shape)
target_run_time = 1000
for method_name in method_name_list:
    for latent_dim in latent_dim_list:
        start_t = time.time()
        params = {"X": org_data_mat, "dataset_name": "Face", "r": latent_dim}
        instance_name = f"NMF_{method_name}"
        method_instance = globals()[instance_name](
            method_name=method_name, params=params
        )
        method_instance.run_within_fixed_time(target_run_time=target_run_time)
        print(f"Finished Method {method_name} in {time.time()- start_t} seconds")
# nmf_enmf = NMF_ENMF(method_name=method_name, params=params)
