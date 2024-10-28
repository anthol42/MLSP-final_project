config_format = {
    "data":{
        "batch_size": int,
        "shuffle": bool,
        "mode": str,
        "p_quant": int,
        "window_len": int,
        "num_workers": int,
    },
    "training":{
        "num_epochs": int,
        "lr": float,
        "min_lr": float,
        "weight_decay": float,
    },
    "model":{
        "model_dir": "opath",
        "model_name": "opath",
        "dropout2d": float,
        "dropout": float
    },
    "scheduler":{
        "n_iter_restart": int,
        "factor_increase": int,
        "min_lr": float
    }
}