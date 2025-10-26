import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def imbalance_measure(column, how="entropy"):
    column = column[column != 2480]
    
    # 1. Gini Coefficient (0,1) = (불균형, 균형)
    if how == "gini": 
        freq = column.value_counts(normalize=True)
        return 1 - sum(freq**2)
    
    # 2. Entropy (불균형, 균형)
    elif how == "entropy":
            freq = column.value_counts(normalize=True)
            return -sum(freq * np.log2(freq))
        
    # 3. Ratio of Largest Class to Smallest Class (1, 무한대) = (균형, 불균형)
    elif how == "ratio": 
        freq = column.value_counts()
        return freq.max() / freq.min()
    
    # 4. Variance of Class Proportions (균형, 불균형)
    elif how == "variance_proportions": 
        freq = column.value_counts(normalize=True)
        return freq.var()    
    
def calculate_skewness(column):
    column = column[column != 0]    
    return abs(skew(column))


def analyze_data_imbalance(data_path):
    try:
        # Load metadata and datasets
        info = pd.read_pickle(os.path.join(data_path, "info_dict.pickle"))
        categorical_df = pd.read_csv(os.path.join(data_path, "original_df.csv")).iloc[:, :info["cat_col_num"]]
        numerical_data = np.load(os.path.join(data_path, "unnormalized_input_ids.npy"))[:, info["cat_col_num"]:info["cat_col_num"]+info["num_col_num"]]
        column_name = pd.read_csv(os.path.join(data_path, "original_df.csv")).iloc[:, info["cat_col_num"]:info["cat_col_num"]+info["num_col_num"]].columns
        numerical_df = pd.DataFrame(numerical_data, columns=column_name)

        # Evaluate class imbalance for categorical data using different methods
        methods = ["gini", "entropy", "variance_proportions"]
        sorted_columns_by_method = {}

        for method in methods:
            imbalance_scores = categorical_df.apply(imbalance_measure, how=method)
            
            # Depending on the method, we might want to sort in ascending or descending order
            ascending = True if method == "variance_proportions" else False
            sorted_columns = imbalance_scores.sort_values(ascending=ascending).index.tolist()
            sorted_columns_by_method[method] = sorted_columns

        # Display the results
        imbalance_df_cat = pd.DataFrame(sorted_columns_by_method)
        print("Categorical Imbalance Analysis:")
        print(imbalance_df_cat.head(), "\n")
        print(imbalance_df_cat.tail(), "\n")

        # Evaluate imbalance for numerical data using standard deviation as a measure
        imbalance_scores_num = numerical_df.apply(calculate_skewness)
        sorted_columns_num = imbalance_scores_num.sort_values(ascending=True).index.tolist()

        # Prepare the dataframe for display
        imbalance_df_num = pd.DataFrame({
            "column": sorted_columns_num,
            "skewness": imbalance_scores_num[sorted_columns_num]
        }).reset_index(drop=True)

        print("Numerical Imbalance Analysis:")
        print(imbalance_df_num.head(), "\n")
        print(imbalance_df_num.tail(), "\n")

        # Save results to CSV
        imbalance_df_cat.to_csv(os.path.join(data_path, "class_imbalance_cat.csv"), index=False)
        imbalance_df_num.to_csv(os.path.join(data_path, "class_imbalance_num.csv"), index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
    return
