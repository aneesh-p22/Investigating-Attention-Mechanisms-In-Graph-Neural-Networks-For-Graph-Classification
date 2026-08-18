CONFIG = {
    "model": "gat",
    "dataset": "MUTAG",
    "hidden_dim": 32,
    "heads": 4,

    "num_folds": 10,
    "inner_val_ratio": 0.1,

    "attention_dropout": 0.0,
    "model_dropout": 0.0,
    "weight_decay": 0.0,
    "negative_slope": 0.2,
    "add_self_loops": True,

    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "split_seeds": [0],
    "training_seeds": [0],

    "search_space": {
        "hidden_dim": [32, 64],
        "learning_rate": [0.001, 0.0005]
    }
}