import numpy as np
import time
import os
from nmf_algos.utils.utils import  load_audio_data, audio_preprocess
from nmf_algos import NMF_ENMF, NMF_HALS, NMF_AOADMM, NMF_MUL, NMF_GRADMUL, NMF_ALS

method_name_list = ["HALS", "MUL","AOADMM", "GRADMUL", "ALS", "ADMM"]
latent_dim_list = [20, 40, 80, 100]
rerun_times = 5

# There are 18000 audios in the training dataset, with 8000 feature per audio.
project_dir = os.path.join(os.getcwd())
f_path = os.path.join(project_dir, "Dataset/audiomnist.npy.npz")
data_mat, data_labels = load_audio_data(f_path)

target_run_time = 600 
dataset_name = "Audiomnist"
print("Loaded data with shape: ", data_mat.shape)
for latent_dim in latent_dim_list:
    for method_name in method_name_list:
        data_mat, data_labels = load_audio_data(f_path)
        data_mat = audio_preprocess(data_mat)
        start_t = time.time()
        params = {"X": data_mat, "dataset_name": dataset_name, "r": latent_dim, "rerun_times":rerun_times}
        instance_name = f"NMF_{method_name}"
        method_instance = globals()[instance_name](method_name=method_name, params=params)
        method_instance.run_within_fixed_time(target_run_time=target_run_time)
        print(f"Finished Method {method_name} in {time.time()- start_t} seconds")
