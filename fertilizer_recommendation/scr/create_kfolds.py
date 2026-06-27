import pandas as pd
from sklearn import model_selection


df = pd.read_csv("../input/fertilizer_recommendation.csv")

df.columns = df.columns.str.lower()

df["kfold"] = -1

# Map fertilizer labels (strings) to integer classes
mapping = {
    "MOP": 0,
    "Urea": 1,
    "Zinc Sulphate" : 2, 
    "Compost": 3, 
    "NPK": 4, 
    "DAP": 5, 
    "SSP": 6
}


# Apply mapping to convert fertilizer names
df["recommended_fertilizer"] = df["recommended_fertilizer"].astype(object)
df.loc[:, "recommended_fertilizer"] = df["recommended_fertilizer"].map(mapping)

#Extract target variable as integer type
y = df["recommended_fertilizer"].astype(int)

# Create Stratified K-Folds
kfd = model_selection.StratifiedKFold(n_splits=5)

#Assign fold numbers
for f, (t_, v_) in enumerate(kfd.split(X=df, y =y)):
    df.loc[v_, "kfold"] = f

# Save the new dataset with fold assignments
df.to_csv("../input/fertilizer_data_kfolds.csv", index=False)

