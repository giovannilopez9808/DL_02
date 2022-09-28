def get_params() -> dict:
    params = {
        "path models": "../Models",
        "path data": "../Data",
        "log path": "../log",
        "dataset": {
            "train": {
                "image_size": (256, 256),
                "batch_size": 1,
                "shuffle": True,
                "labels": None,
            },
            "test": {
                "image_size": (256, 256),
                "batch_size": 1,
                "shuffle": False,
                "labels": None,
            }
        },
        "VAE": {
            "input_dim": (256, 256, 3),
            "r_loss_factor": 1000,
            "latent_dim": 300,
            "summary": False,
        },
        "epochs": 15,
    }
    return params
