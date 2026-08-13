CONFIG = {
    "model": "gin",
    "dataset": "MUTAG",

    "num_folds": 10,
    "inner_val_ratio": 0.1,

    "model_dropout": 0.0,
    "weight_decay": 0.0,

    "hidden_dim": 32,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "split_seeds": [0, 1, 2, 3, 4],
    "training_seeds": [0, 1, 2, 3, 4]
}