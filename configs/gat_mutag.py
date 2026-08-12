CONFIG = {
    "model": "gat",
    "dataset": "MUTAG",
    "hidden_dim": 32,
    "heads": 4,

    "train_ratio": 0.8,
    "val_ratio": 0.1,

    "attention_dropout": 0.0,
    "negative_slope": 0.2,
    "add_self_loops": True,

    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "seeds": [0, 1, 2, 3, 4]
}