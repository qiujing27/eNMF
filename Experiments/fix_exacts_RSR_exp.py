import numpy as np
import os
from multiprocessing import Pool

from nmf_algos.utils.utils import load_data_basedon_proto, fetch_factors_from_result_path
from nmf_algos import NMF_ENMF


project_dir = os.path.join(os.getcwd())
# repeat the experment only 1 time
rerun_times = 1
proto_path = os.path.join(project_dir, "Experiments/configs/exact_data_algo_RSR.json")
dataset_configs = load_data_basedon_proto(proto_path, mode="exactDatasets").exact_dataset
print(dataset_configs)

for dataset_config in dataset_configs[:1]:
    f_path = os.path.join(project_dir, dataset_config.data_dir, dataset_config.data_path)
    print(dataset_config.data_path)
    org_data_mat = fetch_factors_from_result_path(f_path, f_type="exacts", key_list=["X"])
    print("org_data_mat shape", org_data_mat.shape)
    latent_dim = int(dataset_config.method_config[0].latent_dim)
    #print("latent dim:", latent_dim)
    params = {"X": org_data_mat, "dataset_name":dataset_config.name, "r": latent_dim}
    admm_config = {"rho": 5, "epsilon":10 ** (-4), "max_iter": 10000, "tau_inc":1.1, "tau_dec":1.1, "num_steps":10, "hals_rounds":1, "rerun_times":rerun_times}
    params.update(admm_config)
    nmf_enmf = NMF_ENMF(params=params)
    nmf_enmf.basic_run()
    #nmf_enmf.run_within_fixed_time(target_run_time=target_t)


