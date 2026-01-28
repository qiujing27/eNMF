def get_config(stage, dataset_name):
    """
    Stage: 1. first rotation (SVD_rotation)
    2. ascent step additional rotation
    3. descent part rotation? (ONGOING)
    """
    if stage == "SVD_rotation":
        rho = 1.6
        epsilon = 10 ** (-3)
        max_iter = 2000
        tau_inc = 1.1
        tau_dec = 0.998
        mu = 2
        rho_mode = 0
    if stage == "SVD_rotation_syn":
        # rho = 50
        rho = 10
        epsilon = 10 ** (-4)
        max_iter = 4000
        tau_inc = 1.1
        tau_dec = 1.1
        mu = 2
        rho_mode = 0
    if stage == "SVD_rotation_face":
        # rho = 50
        rho = 10
        epsilon = 10 ** (-3)
        max_iter = 2000
        tau_inc = 1.1
        tau_dec = 0.998
        mu = 2
        rho_mode = 0
    if stage == "SVD_rotation_verb":
        # rho = 50
        rho = 5
        epsilon = 10 ** (-4)
        max_iter = 20000
        tau_inc = 1.1
        tau_dec = 1.1
        mu = 2
        rho_mode = 0

    if stage == "SVD_rotation_add":
        rho = 1.6
        epsilon = 10 ** (-6)
        max_iter = 20000
        tau_inc = 1.1
        tau_dec = 0.998
        mu = 2
        rho_mode = 0

    if stage == "Ascent_rotation":
        rho = 1.6
        epsilon = 10 ** (-3)
        max_iter = 1000
        tau_inc = 1.1
        tau_dec = 0.998
        mu = 2
        rho_mode = 0

    config_dict = {}
    config_dict["rho"] = rho
    config_dict["epsilon"] = epsilon
    config_dict["max_iter"] = max_iter
    config_dict["tau_inc"] = tau_inc
    config_dict["tau_dec"] = tau_dec
    config_dict["mu"] = mu
    config_dict["rho_mode"] = rho_mode

    return config_dict


def get_config_ascent(method, dataset_name):
    if method == "vanilla":
        config_dict = {}
        tol_asc = 0.2
        inner_iter_asc = 2
        step_percent = 0.001

        config_dict["tol_asc"] = tol_asc
        config_dict["inner_iter_asc"] = inner_iter_asc
        config_dict["asc_step_per"] = step_percent
        return config_dict
    if dataset_name == "verb":
        config_dict = {}
        tol_asc = 0.2
        inner_iter_asc = 1
        step_percent = 0.01
        config_dict["tol_asc"] = tol_asc
        config_dict["inner_iter_asc"] = inner_iter_asc
        config_dict["asc_step_per"] = step_percent
        return config_dict
