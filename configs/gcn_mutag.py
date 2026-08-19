CONFIG = {
    "model": "gcn",
    "dataset": "MUTAG",

    "num_folds": 10,
    "inner_val_ratio": 0.1,

    "split_seeds": [0],
    "training_seeds": [0],
    "tuning_seeds": [0, 1, 2],

    "model_dropout": 0.0,
    "weight_decay": 0.0,

    "hidden_dim": 32,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "split_seeds": [0, 1, 2, 3, 4],
    "training_seeds": [0, 1, 2, 3, 4],

    "search_space": {
        "hidden_dim": [32, 64],
        "learning_rate": [0.001, 0.0005],
        "model_dropout": [0.0, 0.5],
        "weight_decay": [0.0, 0.0005]
    },
}