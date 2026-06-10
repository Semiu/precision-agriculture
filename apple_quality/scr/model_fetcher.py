from sklearn import linear_model
from sklearn import ensemble
from sklearn import model_selection

import numpy as np


import warnings

warnings.filterwarnings(
    "ignore",
    message="'n_jobs' > 1 does not have any effect"
)


# Baseline model
models = {
    "lr": linear_model.LogisticRegression(n_jobs=-1),
    "rf": ensemble.RandomForestClassifier(n_jobs=-1)
}

# hyperparameter tuner 
param_grids = {
    "lr": {
        "penalty": ["l1", "l2"],
        "C": [0.1, 1, 10],
        "solver": ["liblinear"]   
    },
    "rf": {
        "n_estimators": np.arange(100, 500, 100),
        "max_depth": np.arange(1, 12),
        "criterion": ["gini", "entropy"]
    }
}

def get_grid_search(model_name):
    return model_selection.GridSearchCV(
        estimator=models[model_name],
        param_grid=param_grids[model_name],
        scoring="roc_auc",
        cv=5,
        verbose=10
    )


def get_random_search(model_name):
    return model_selection.RandomizedSearchCV(
        estimator=models[model_name],
        param_distributions=param_grids[model_name],
        n_iter=10,
        scoring="roc_auc",
        cv=5,
        verbose=10
    )
