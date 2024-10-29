config_format = {
    "data":{
        "batch_size": int,
        "shuffle": bool,
        "mode": str,
        "p_quant": int,
        "window_len": int,
        "num_workers": int,
        "random_seed": int
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
        "dropout": float
    }
}