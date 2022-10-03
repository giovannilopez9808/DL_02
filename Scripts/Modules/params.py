def get_params() -> dict:
    params = {
        "path checkpoint": "../Checkpoint",
        "path graphics": "../Graphics",
        "path models": "../Models",
        "path data": "../Data",
        "path log": "../log",
        "dataset": {
            "train": {
                "image_size": (256, 256),
                "batch_size": 8,
                # "batch_size": 1,
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
        "epochs": 10000,
        # "epochs":0,
    }
    return params
