import os
import glob
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier as cb
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# def train_model(X_train, y_train, model_type, categorical_features_indices, seed):
#     model_dict = {
#         "LR": LogisticRegression(random_state=seed),
#         "RF": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed),
#         "KNN": KNeighborsClassifier(n_neighbors=5),
#         "DT": DecisionTreeClassifier(random_state=seed),
#         "CB": cb(iterations=100, 
#               depth=6, 
#               learning_rate=0.1, 
#               loss_function='Logloss',  
#               random_seed=seed)
#     }
#     model = model_dict[model_type]
#     if model_type == "CB":
#         model.fit(X_train, y_train, cat_features=categorical_features_indices, verbose=False)
#     else:
#         model.fit(X_train, y_train)
    
#     return model


def preprocess_and_fill_missing(X, num_categorical_cols, replace_nulls):
    """
    Preprocess the DataFrame by handling missing values in both categorical and numerical columns.
    """
    # Handle categorical columns: fill missing values with '-99' and convert to string
    X.iloc[:, :num_categorical_cols] = X.iloc[:, :num_categorical_cols].fillna('-99').astype(str)
    # Handle numerical columns: fill missing values with '-99' and convert to float
    X.iloc[:, num_categorical_cols:] = X.iloc[:, num_categorical_cols:].fillna(-99).astype(float)
    # Optionally replace '-99' in numerical columns with NaN for certain models that handle NaNs internally
    if replace_nulls:
        X.iloc[:, num_categorical_cols:] = X.iloc[:, num_categorical_cols:].replace(-99, float("NaN"))
    return X
    

def evaluate_TSTR(X_train, X_test, y_train, y_test, num_categorical_cols, seed, replace_nulls=True, extract_feat_importance=None):
    columns_to_exclude = ['AKI_Cr2_Days_after_Op_start', 'AKI_Max_Stage', 'AKI_first_Stage', 'CA_AKI_2W', 'O_AKI', 'O_AKI_criteria1', 'O_AKI_criteria2', 'O_AKI_stage', 'O_CRRT_7', 'O_CRRT_90', 'O_Critical_AKI_7', 'O_Critical_AKI_90', 'O_Death_7', 'O_Death_90', 'O_HD_7', 'O_HD_90', 'O_RRT_7', 'O_RRT_90', 'AKI_Cr2_Cr_diff', 'AKI_Cr2_Value', 'AKI_Max_Value', 'AKI_first_Value']

    # Validate that none of the excluded columns are present in the training and testing data
    if any(col in X_train.columns for col in columns_to_exclude):
        raise ValueError("Input dataframes must not include outcome or specific excluded columns.")

    X_train = preprocess_and_fill_missing(X_train, num_categorical_cols, replace_nulls)
    X_test = preprocess_and_fill_missing(X_test, num_categorical_cols, replace_nulls)

    # Determine indices of categorical features for CatBoost
    categorical_var = X_train.columns[:num_categorical_cols]
    categorical_features_indices = [list(X_train.columns).index(i) for i in categorical_var]

    # Initialize model based on the type (assuming CatBoost here, extendable to other types)
    model = cb(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', random_seed=seed)
    model.fit(X_train, y_train, cat_features=categorical_features_indices, verbose=False)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate AUROC
    auroc = roc_auc_score(y_test, y_pred_prob)
    # Calculate AUPRC
    auprc = average_precision_score(y_test, y_pred_prob)

    # Extract feature importances
    if extract_feat_importance:
        feature_importances = model.get_feature_importance()
        feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
        feature_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

        # Save feature importances to a CSV file
        feature_importances_df.to_csv(extract_feat_importance, index=False)

    # Additional performance metric could be included here such as AUPRC
    return model, y_pred_prob, round(auroc, 4), round(auprc, 4)
    