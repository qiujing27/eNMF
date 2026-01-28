import numpy as np
import os
import multiprocessing
from multiprocessing import Pool
from nmf_algos.utils.utils import (
    load_data_basedon_proto,
    fetch_factors_from_result_path,
)
from nmf_algos import NMF_ALS
from nmf_algos import NMF_ENMF

method_name_list = ["ALS", "AOADMM"]  # "HALS, "ENMF"
project_dir = os.path.join(os.getcwd(), "ENMF")
## Exact dataset
# proto_path = os.path.join(project_dir, "Data/exact_data_algo_exp.json")
# dataset_config = load_data_basedon_proto(proto_path, mode="exactDatasets").exact_dataset
## Synthetic datasets with different sparsities
# proto_path = os.path.join(project_dir, "Data/syn_data_algo_exp.json")
# dataset_configs = load_data_basedon_proto(proto_path, mode="synDatasets").real_dataset

proto_path = os.path.join(project_dir, "Data/exact_data_algo_RSR.json")
dataset_configs = load_data_basedon_proto(
    proto_path, mode="exactDatasets"
).exact_dataset
print(dataset_configs)
for dataset_config in dataset_configs:
    f_path = os.path.join(
        project_dir, dataset_config.data_dir, dataset_config.data_path
    )
    print(dataset_config.data_path)
    org_data_mat = fetch_factors_from_result_path(
        f_path, f_type="exacts", key_list=["X"]
    )
    print("org_data_mat shape", org_data_mat.shape)
    latent_dim = int(dataset_config.method_config[0].latent_dim)
    # print("latent dim:", latent_dim)
    params = {"X": org_data_mat, "dataset_name": dataset_config.name, "r": latent_dim}
    nmf_enmf = NMF_ENMF(params=params)
    nmf_enmf.basic_run()
    # nmf_enmf.run_within_fixed_time(target_run_time=target_t)

# latent_dim = 500
# target_t = 600
# for dataset_config in dataset_configs:
#     f_path = os.path.join(project_dir, dataset_config.data_dir, dataset_config.data_path)
#     #fetch_factors_from_result_path(f_path, key_list=["X"])
#     #print(org_data_mat.shape)
#     print( dataset_config.data_path)
#     for latent_dim in [100, 200, 300, 400]:
#         org_data_mat = fetch_factors_from_result_path(f_path)
#         params = {"X": org_data_mat, "dataset_name":dataset_config.name, "r": latent_dim}
#         nmf_enmf = NMF_ENMF(params=params)
#         # #nmf_enmf.basic_run()
#         nmf_enmf.run_within_fixed_time(target_run_time=target_t)


# nmf_enmf = NMF_ENMF(params=params)
# nmf_enmf.basic_run()
# nmf_enmf = NMF_ENMF(params=params)
# nmf_enmf.basic_run()
# target_t =  100
# nmf_als = NMF_ALS( params=params)
# nmf_als.run_within_fixed_time(target_run_time=target_t)


# print(real_dataset.name)
#     #params = {"X": org_data_mat, "r": 20, "save_dir": "", "iter_save_dir": }
#     latent_dims = [10, 20, 40, 80, 100]
#     target_time = [3600, 3600, 3600, 3600, 3600]
#     for latent_dim, target_t in zip(latent_dims, target_time):
#         params = {"X": org_data_mat, "dataset_name": "Verb", "r": latent_dim, "rerun_times": 1}
#         nmf_als = NMF_ALS(method_name=method_name, params=params)
#         nmf_als.run_within_fixed_time(target_run_time=target_t)

# with Pool() as pool:
#     # call the same function with different data in parallel
#     column_idx = 0
#     #for result in pool.imap(scd_ls_update_column_parallel, [{"Aj": A[:,j], "W": W, "Hj": H[:, j],"row_mask":known_mask[:, j],"beta": beta, "inner_rel_tol": inner_rel_tol, "inner_max_iter":inner_max_iter} for j in range(n)]):
#     for result in pool.imap(mul_ls_update_column_parallel, [(A[:,j],  W, H[:, j], known_mask[:, j], beta,  inner_rel_tol, inner_max_iter) for j in range(n)]):
#         pass
