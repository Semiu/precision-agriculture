
"""
train.py

This script trains and evaluates a machine learning model using 5‑fold
stratified cross‑validation. It supports:

1. Baseline models (Logistic Regression or Random Forest)
2. Hyperparameter optimization using:
      - GridSearchCV
      - RandomizedSearchCV

Workflow:
- Load and preprocess dataset
- Select model based on user input
- Optionally perform hyperparameter tuning
- Create stratified folds
- Train on 4 folds, validate on 1 fold
- Compute Accuracy and AUC for the selected fold

This modular design allows flexible experimentation with different models
and tuning strategies without modifying the core training logic.
"""

import config
import model_fetcher
import argparse
import pandas as pd
from sklearn import model_selection
from sklearn import metrics


def run(fold, model_name, tuner):  
    
    """
    Runs training and evaluation for a single fold.

    Parameters:
    - fold (int): which fold (0-4) to use as validation
    - model_name (str): "lr" for Logistic Regression or "rf" for Random Forest
    - tuner (str): "none", "grid", or "random"

    The function:
    - Loads and preprocesses the dataset
    - Selects the appropriate model (baseline or tuned)
    - Creates stratified folds
    - Trains the model on training folds
    - Evaluates on the validation fold
    """

    
    # Load the training CSV file
    df = pd.read_csv(config.TRAINING_FILE)

    #convert all column names to lowercase for consistency
    df.columns = df.columns.str.lower()

    #mapping dictionary to convert string labels to numeric
    mapping = {"good": 1, "bad": 0 }

    # apply the mapping to the target column
    df.quality = df.quality.map(mapping)

    X = df.drop("quality", axis=1).values
    y = df.quality.values

    if tuner == "grid":
        print("---Using GridSearchCV for hyperparameter tuning")
        grid_model =  model_fetcher.get_grid_search(model_name).fit(X, y)
        model = grid_model.best_estimator_

    elif tuner =="random":
        print("---Using RandomsizeSearchCV for hyperparameter tuning")
        random_model = model_fetcher.get_random_search(model_name).fit(X, y)
        model = random_model.best_estimator_

    else: 
        print("---using Baseline Model (no hyperparameter tuning)")
        model = model_fetcher.models[model_name]


    #Create fold
    # Shuffle the dataset, drop the ID column and index
    df = df.sample(frac=1).drop("a_id", axis=1).reset_index(drop=True)

    #create a new column to store the fold assignments
    df["kfold"] = -1

    #extract target column
    y = df.quality

    # Initialize stratified K-fold object
    kfd = model_selection.StratifiedKFold(n_splits=5)

    # Assign fold numbers to each row 
    for f, (t_, v_) in enumerate(kfd.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    
    # training data = all folds except the one pass in
    df_train = df[df["kfold"]  != fold].reset_index(drop=True)

    # test data = only the selected fold
    df_test = df[df["kfold"] == fold].reset_index(drop=True)

    # split into features (X) and target (y)
    x_train = df_train.drop(["quality", "kfold"], axis=1).values
    y_train = df_train["quality"].values

    x_test = df_test.drop(["quality", "kfold"], axis=1).values
    y_test = df_test["quality"].values


    # Train the model using final model
    model.fit(x_train, y_train)

    # predict class label (0 or 1)
    preds = model.predict(x_test)

    # predict probabilities for class 1
    preds_pro = model.predict_proba(x_test)[:, 1]

    # Compute evaluation metrics
    accuracy = metrics.accuracy_score(y_test, preds)

    # compute AUC score using probabilities
    auc = metrics.roc_auc_score(y_test, preds_pro)

    #print evaluation results for the pass in fold. 
    print(f"Fold {fold} | Model: {model_name} | Tunner: {tuner} | Accuracy: {accuracy} | AUC: {auc}") 


if __name__ == "__main__":

    # Parse comman-line argument 
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str, choices=["lr", "rf"])
    parser.add_argument("--tuner", type=str, choices=["none", "grid", "random"])
    args = parser.parse_args()

    # Run training for the selected fold
    run(fold = args.fold, model_name = args.model, tuner=args.tuner)