def get_params() -> dict:
    params = {
        "path graphics": "../Graphics",
        "path models": "../Models",
        "path data": "../Data",
        "log path": "../log",
        "dataset": {
            "train": {
                "image_size": (256, 256),
                "batch_size": 32,
                "shuffle": True,
                "labels": None,
            },
        },
        "VAE": {
            "input_dim": (256, 256, 3),
            "r_loss_factor": 100000,
            "latent_dim": 150,
            "summary": False,
        },
        "epochs":100,
        # "epochs":10,
    }
    return params
