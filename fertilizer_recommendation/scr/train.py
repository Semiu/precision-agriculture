import features_engine
import model_dispatcher

import argparse

import itertools
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing


from sklearn.datasets import load_diabetes 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.feature_selection import SelectFromModel


# Load dataset with stratified folds
df = pd.read_csv("../input/fertilizer_data_kfolds.csv")

# Apply feature engineering
data = features_engine.feature_engineering(df)
X = data.drop(["recommended_fertilizer", "kfold"], axis=1)
col_names = X.columns
y = data["recommended_fertilizer"]

# initialize the model 
model = RandomForestRegressor()  

# Function: Select important features using SelectFromModel
def get_importance_feature():
    # select from the model 
    sfm = SelectFromModel(estimator=model) 
    X_transformed = sfm.fit_transform(X, y)  

    # see which features were selected 
    support = sfm.get_support()  

    # get feature names 
    features = [x for x, y in zip(col_names, support) if y == True]
    return features

# Function: Return raw feature importance values
def importance_features():
    model.fit(X, y)
    return model.feature_importances_

# Training function for a single fold
def run(fold, model):
    
    # Shuffle dataset for randomness
    data = df.sample(frac=1).reset_index(drop=True)

    # Split into training and validation sets based on fold number
    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)

    x_train = df_train[get_importance_feature()].values
    y_train = df_train["recommended_fertilizer"].values

    x_valid = df_valid[get_importance_feature()].values
    y_valid = df_valid["recommended_fertilizer"].values

    # Retrieve model from model dispatcher
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    pred = clf.predict(x_valid)

    accuracy = metrics.roc_auc_score(y_valid, pred, average="weighted")
    print("Fold:", fold, "Accuracy:", accuracy)

# Command-line execution
# Allows running: python train.py --fold 0 --model rf
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    run(fold=args.fold, model=args.model)