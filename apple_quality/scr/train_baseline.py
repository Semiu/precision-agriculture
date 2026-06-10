import config
import model_fetcher

import argparse


import pandas as pd

from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics


def run(fold, selected_model): 
    """
    the function runs training and evaluation for a single fold in a 5-fold
    stratified cross-validation setup.

    Steps performed:
    - Load dataset
    - Clean column names
    - Map target labels (good → 1, bad → 0)
    - Shuffle data and remove ID column
    - Create stratified folds
    - Split into training and validation sets
    - Train the selected model
    - Evaluate using Accuracy and AUC

    NOTE:
    This is the baseline implementation.
    A more advanced and optimized version of the model
    will be developed in a separate script.
    """
    # Load the training CSV file
    df = pd.read_csv(config.TRAINING_FILE)

    #convert all column names to lowercase for consistency
    df.columns = df.columns.str.lower()

    #mapping dictionary to convert string labels to numeric
    mapping = {
        "good": 1,
        "bad": 0
    }

    # apply the mapping to the target column
    df.quality = df.quality.map(mapping)

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

    # Initialize Logistic Regression model
    model = model_fetcher.models[selected_model]

    # Train the model
    model.fit(x_train, y_train)

    # predict class label (0 or 1)
    preds = model.predict(x_test)

    # predict probabilities for class 1
    preds_pro = model.predict_proba(x_test)[:, 1]

    # Compute accuracy score
    accuracy = metrics.accuracy_score(y_test, preds)

    # compute AUC score using probabilities
    auc = metrics.roc_auc_score(y_test, preds_pro)

    #print evaluation results for the pass in fold. 
    print(f"Fold {fold} accuracy: {accuracy} and auc: {auc}") 


if __name__ == "__main__":

    # Parse comman-line argument for fold number
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--selected_model", type=str)
    args = parser.parse_args()

    # Run training for the selected fold
    run(fold = args.fold, selected_model = args.selected_model)










