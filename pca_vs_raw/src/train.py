import argparse
import os



import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.decomposition import PCA

import config
import model_dispatcher


def run(fold, model, n_components = None):
    # Read data wit folds
    df = pd.read_csv(config.TRAINING_FILE)

    #drop the target and kfold columns
    X = df.drop(['target', 'kfold'], axis=1)
    y = df[["target", "kfold"]]

    #Apply PCA only if n_components is provided
    if n_components is not None: 
        data_pca = PCA(n_components=n_components)
        data_pca = data_pca.fit_transform(X)
        data = pd.DataFrame(data_pca)
    else: 
        data = X.copy()
    
    #remerge the target and kfold column
    data[["target","kfold"]] = y

    # Training data is where fold is not equal to provided fold
    df_train = data[data.kfold != fold].reset_index(drop=True)
    df_valid = data[data.kfold == fold].reset_index(drop=True)

    # drop the target column in the df_train and df_valid variable and convert it to numpy
    # take the target column as y_train and y_valid variable
    x_train = df_train.drop(["target","kfold"], axis=1).values
    y_train = df_train["target"].values

    x_valid = df_valid.drop(["target","kfold"], axis=1).values
    y_valid = df_valid["target"].values

    # initialize simple decison tree classifier from sklearn
    clf = model_dispatcher.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create prediction for validation data
    preds = clf.predict(x_valid)
    

    #Probability for ROC-AUC
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x_valid)

        #Binary classification
        if probs.shape[1] == 2:
            probs = probs[:, 1]


    # calculate and print metrics 
    accuracy = metrics.accuracy_score(y_valid, preds)
    f_1_score =metrics.f1_score(y_valid, preds, average="weighted")
    precision = metrics.precision_score(y_valid, preds, average="weighted")
    auc = metrics.roc_auc_score(y_valid, probs, multi_class="ovr") 

    print(f"This run uses {n_components} PCA components with the {model} model, based on fold {fold}.")
    print("Accuracy score is most reliable when the data is balanced or the classes are evenly distributed.")
    print(f"Fold {fold} | " f"Accuracy={accuracy:.4f} | " 
    f"F1 score={f_1_score:.4f} | " 
    f"Precision={precision:.4f} | "
    f"AUC={auc:.4f} | "  
    ) 

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))


if __name__ == "__main__":
    #intialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    #add the different arguments you need and their type
    parser.add_argument("--fold", type = int)
    parser.add_argument("--model", type = str)
    parser.add_argument("--n_components", type = int)

    # read the arugment from the command line
    args = parser.parse_args()

# run the fold specified by command lines arguments
run(fold=args.fold, model=args.model, n_components=args.n_components)
    