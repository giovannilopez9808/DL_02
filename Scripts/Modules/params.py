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
                "batch_size": 1,
                "shuffle": True,
                "labels": None,
            },
        },
        "VAE": {
            "input_dim": (256, 256, 3),
            "r_loss_factor": 1000000,
            "latent_dim": 600,
            "summary": False,
        },
        "epochs": 50,
    }
    return params
