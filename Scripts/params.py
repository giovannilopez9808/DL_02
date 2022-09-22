def get_params() -> dict:
    params = {
        "path models": "../Models",
        "path data": "../Data",
        "log path": "../log",
        "dataset": {
            "train": {
                "image_size": (256, 256),
                "batch_size": 9,
                "shuffle": True,
                "labels": None,
            }
        },
        "VAE": {
            "input_dim": (256, 256, 3),
            "r_loss_factor": 100000,
            "latent_dim": 150,
        },
        "epochs": 400,
    }
    return params
