config_format = {
    "data":{
        "batch_size": int,
        "shuffle": bool,
        "mode": str,
        "p_quant": int,
        "window_len": int,
        "num_workers": int,
        "random_seed": int,
        "space_between": int,
        "enlarge_factor": int,
        "interpolation_factor": int,
        "offset": int,
        "plt_fig": bool,
        "group_size": int,
    },
    "training":{
        "num_epochs": int,
        "lr": float,
        "min_lr": float,
        "weight_decay": float,
    },
    "model":{
        "model_dir": "opath",
        "name": "opath",
        "dropout": float
    }
}