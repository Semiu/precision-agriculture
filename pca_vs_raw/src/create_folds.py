import pandas as pd
from sklearn import model_selection


# the if statment is making sure that this file cannot be imported
if __name__ == "__main__":

    # data is in CSV file
    #def make_fold():
    df = pd.read_csv('../input/train_mnist.csv')

    #create a new column 'kfold' and fill it with -1
    df['kfold'] = -1

    #shuffle the rows of the data and drop off the index value
    df = df.sample(frac=1).reset_index(drop=True)

    #separate target column
    y = df["target"].values

    #initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    #fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    #save the new csv with kfold column in input directory
    df.to_csv('../input/train_mnist_folds.csv', index=False)    