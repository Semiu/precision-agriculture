import pandas as pd
import itertools

from sklearn import preprocessing




def feature_engineering(df):
    """
    Perform feature engineering on the fertilizer dataset.
    Includes:
    - Categorical pair combinations
    - One-hot encoding
    - Scaling numerical features
    - Polynomial feature generation
    - Final dataset assembly
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Identify categorical and numerical columns
    cat_cols = ["soil_type", "crop_type", "crop_growth_stage", "season", "irrigation_type", "previous_crop", "region"]

    # Numerical columns exclude categorical + target + fold column
    num_cols = [col for col in df.columns if col not in cat_cols and col not in ["recommended_fertilizer", "kfold"]]

    # Create pairwise combinations of categorical variables
    combi = list(itertools.combinations(cat_cols, 2))
    
    # Categorical variable combination
    for c1, c2 in combi: 
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)

     # Recompute categorical variables (including new combinations to include the new categorical variable combination
    cat_var = [col for col in df.columns if col not in num_cols and col not in ["recommended_fertilizer", "kfold"]]

    # One-Hot Encode all categorical variables
    ohen = preprocessing.OneHotEncoder()
    ohen.fit(df[cat_var])
    encoded = ohen.transform(df[cat_var]).toarray()
    columns = ohen.get_feature_names_out(cat_var)

    df_encoded = pd.DataFrame(encoded, columns=columns)

    #Scale numeric features
    scaler = preprocessing.StandardScaler()
    scaled_num = scaler.fit_transform(df[num_cols])

    # Polynomail features (degree = 3)
    pf = preprocessing.PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

    pf.fit(scaled_num)
    poly_features =  pf.transform(scaled_num)

    df_poly_features = pd.DataFrame(poly_features, columns=pf.get_feature_names_out(num_cols))

    # Combine: Polynomial numerical features, One-hot encoded categorical features and Target + fold columns
    df_combined = pd.concat([df_poly_features, df_encoded, df[["recommended_fertilizer", "kfold"]]], axis=1)
    
    return df_combined


