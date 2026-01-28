import numpy as np
import time
import os
from nmf_algos.utils.utils import (
    load_data_basedon_proto,
    load_data_matrix,
    audio_preprocess,
    load_audio_data,
)
from nmf_algos import NMF_ENMF, NMF_HALS, NMF_AOADMM, NMF_MUL, NMF_GRADMUL, NMF_ALS

# TODO: ENMF multiple runs. Check verbose option, only print save dir here.
# Run to target error, if not reached with three hour, stop anyway.

project_dir = os.path.join(os.getcwd())

dataset_name = "Audiomnist"
f_path = os.path.join(project_dir, "Dataset/audiomnist.npy.npz")
latent_dim_list = [10, 20, 40, 80, 100]
target_run_time = 60
# target_run_time = 600
# target_run_time = 1000
rerun_times = 3
method_name_list = ["ENMF"]
for method_name in method_name_list:
    for latent_dim in latent_dim_list[:1]:
        start_t = time.time()
        dataset_new_name = f"{dataset_name}"
        data_mat, data_labels = load_audio_data(f_path)
        data_new_mat = audio_preprocess(data_mat)
        print("data_new_mat min val", np.min(data_new_mat))
        admm_config = {
            "rho": 5,
            "epsilon": 10 ** (-4),
            "max_iter": 2000,
            "tau_inc": 1.1,
            "tau_dec": 1.1,
            "num_steps": 10,
            "hals_rounds": 1,
            "rerun_times": rerun_times,
        }
        params = {"X": data_new_mat, "dataset_name": dataset_new_name, "r": latent_dim}
        params.update(admm_config)
        instance_name = f"NMF_{method_name}"
        method_instance = globals()[instance_name](
            method_name=method_name, params=params
        )
        # method_instance.basic_run()
        method_instance.run_within_fixed_time(target_run_time=target_run_time)
        print(method_instance.intermediate_result_dict)
        print(f"Finished Method {method_name} in {time.time()- start_t} seconds")
# nmf_enmf = NMF_ENMF(method_name=method_name, params=params)
