import pandas as pd
import os
import glob
from tqdm import tqdm
import sys
import pickle
import numpy as np
from scipy.special import inv_boxcox

from mmtg.utils.tstr import *
from mmtg.utils.stat_test import StatAnalysis

def create_label_table(input_path, info, seed, dataset_type):    
    input_ids = np.load(os.path.join(input_path, "unnormalized_input_ids.npy"))
    null_type_ids = np.load(os.path.join(input_path, "null_type_ids.npy"))
    fold = pd.read_csv(os.path.join(input_path, "fold", f"snuh_{seed}_fold_split.csv"))

    # Ensure shapes of input_ids and null_type_ids match
    assert null_type_ids.shape == input_ids.shape, "Shape of null_type_ids must match input_ids"

    # Determine which data subset to use
    split  = {
        "test": fold.fold.values == 0,
        "valid": fold.fold.values == 2,
        "total": np.ones(len(fold), dtype=bool),  # Select all
        "trainvalid": fold.fold.values != 0
    }[dataset_type]

    # Create DataFrame with appropriate columns up to num_table
    num_table = info["cat_col_num"] + info["num_col_num"]
    df = pd.DataFrame(data=input_ids[:, :num_table], columns=info["column_order"])

    # Select data based on the split
    df = df.iloc[split]
    input_ids = input_ids[split]
    null_type_ids = null_type_ids[split]
    
    # Apply NaNs to categorical and numerical columns based on null_type_ids
    cat_col_num = info["cat_col_num"]
    df.iloc[:, :cat_col_num] = df.iloc[:, :cat_col_num].where(~null_type_ids[:, :cat_col_num])
    df.iloc[:, cat_col_num:] = df.iloc[:, cat_col_num:].where(~null_type_ids[:, cat_col_num:])

    # Assertions to check the consistency of NaN application
    cat_null_ids = info["cat_null_ids"]    
    assert (null_type_ids[:, :cat_col_num].sum() == df.iloc[:, :cat_col_num].isnull().sum().sum() == (input_ids[:, :cat_col_num] == cat_null_ids).sum()), \
        "Mismatch in NaN assignment for categorical columns"
    assert (null_type_ids[:, cat_col_num:].sum() == df.iloc[:, cat_col_num:].isnull().sum().sum()), \
        "Mismatch in NaN assignment for numerical columns"
    return df

def extract_columns_for_pred(compatible_col_path='compatible_columns.pickle'):

    def find_compatible_columns(df1, df2, cols1, cols2):        
        common_cols = set(cols1).intersection(cols2)
        common_df1 = df1[common_cols].dropna(axis=1, how='all')
        common_df2 = df2[common_cols].dropna(axis=1, how='all')
        compatible_cols = set(common_df1.columns).intersection(common_df2.columns)
        return list(compatible_cols)

    if not os.path.isfile(compatible_col_path):
        hosp2data = {}
        for hosp in ['SNUH', 'SNUBH', 'BRMH', 'KNUH', 'CAUH', 'KUMC', 'AMC', 'BKC']:
            input_path = f"/data/20240524/{hosp}/znorm"
            
            # Load config from JSON (new format) or fall back to pickle (old format)
            import json
            config_json_path = os.path.join(input_path, "config.json")
            if os.path.exists(config_json_path):
                with open(config_json_path, 'r') as f:
                    info = json.load(f)
            else:
                # Fall back to pickle for backward compatibility
                info = pd.read_pickle(os.path.join(input_path, "info_dict.pickle"))
            hosp2data[hosp] = {
                "df": create_label_table(input_path, info, seed=0, dataset_type="total"),
                "total_col": info["column_order"],
                "categorical_col": info["column_order"][:info["cat_col_num"]],
                "numerical_col": info["column_order"][info["cat_col_num"]:]
            }

        # Generate 7x7 DataFrames for total, categorical, and numerical compatible columns
        default_dismatched_indices = ["Study", "Op_Code"]
        columns_to_exclude = ['AKI_Cr2_Days_after_Op_start', 'AKI_Max_Stage', 'AKI_first_Stage', 'CA_AKI_2W', 'O_AKI', 'O_AKI_criteria1', 'O_AKI_criteria2', 'O_AKI_stage', 'O_CRRT_7', 'O_CRRT_90', 'O_Critical_AKI_7', 'O_Critical_AKI_90', 'O_Death_7', 'O_Death_90', 'O_HD_7', 'O_HD_90', 'O_RRT_7', 'O_RRT_90', 'AKI_Cr2_Cr_diff', 'AKI_Cr2_Value', 'AKI_Max_Value', 'AKI_first_Value']

        hospitals = list(hosp2data.keys())
        compatible_counts = {'total': pd.DataFrame(0, index=hospitals, columns=hospitals),
                            'categorical': pd.DataFrame(0, index=hospitals, columns=hospitals),
                            'numerical': pd.DataFrame(0, index=hospitals, columns=hospitals)}

        hosp2hosp = {}
        for i, hosp1 in enumerate(hospitals):
            hosp2hosp[hosp1] = {}
            for j, hosp2 in enumerate(hospitals):
                hosp2hosp[hosp1][hosp2] = {}
                for col_type in ['total', 'categorical', 'numerical']:
                    if i != j:
                        cols1 = hosp2data[hosp1][f"{col_type}_col"]
                        cols2 = hosp2data[hosp2][f"{col_type}_col"]
                        
                        cols1 = [col for col in cols1 if col not in columns_to_exclude+default_dismatched_indices]
                        cols2 = [col for col in cols2 if col not in columns_to_exclude+default_dismatched_indices]
                        
                        compatible_cols = find_compatible_columns(hosp2data[hosp1]['df'], hosp2data[hosp2]['df'], cols1, cols2)
                        compatible_counts[col_type].iloc[i, j] = len(compatible_cols)
                        hosp2hosp[hosp1][hosp2][col_type] = compatible_cols
                    else:
                        cols1 = hosp2data[hosp1][f"{col_type}_col"]
                        cols1 = [col for col in cols1 if col not in columns_to_exclude]
                        compatible_counts[col_type].iloc[i, j] = len(cols1)
                        hosp2hosp[hosp1][hosp2][col_type] = cols1

        with open(compatible_col_path,'wb') as fw:
            pickle.dump(hosp2hosp, fw)

    else:
        with open(compatible_col_path, 'rb') as fr:
            hosp2hosp = pickle.load(fr)

    return hosp2hosp

def calculate_categorical_statistics(cat_df):
    """Calculate statistics for categorical data."""
    cat_analysis_df = pd.DataFrame()
    most_frequent = cat_df.mode().iloc[0]
    most_frequent_ratio = cat_df.apply(lambda x: list(x).count(most_frequent[x.name])/len(x))
    least_frequent = cat_df.apply(lambda x: x.value_counts().idxmin())
    least_frequent_ratio = cat_df.apply(lambda x: list(x).count(least_frequent[x.name])/len(x))
    null_ratio = cat_df.isnull().sum() / len(cat_df)

    cat_analysis_df['most_frequent_ratio'] = most_frequent_ratio
    cat_analysis_df['least_frequent_ratio'] = least_frequent_ratio
    cat_analysis_df['null_ratio'] = null_ratio
    
    return cat_analysis_df

def calculate_numerical_statistics(num_df):
    """Calculate statistics for numerical data."""
    num_analysis_df = pd.DataFrame()
    
    stats = ['mean', 'std', 'min', 'max']
    for stat in stats:
        num_analysis_df[stat] = getattr(num_df, stat)()

    quantiles = {'first_quartile': 0.25, 'median': 0.5, 'third_quartile': 0.75}
    for name, q in quantiles.items():
        num_analysis_df[name] = num_df.quantile(q)
    
    null_ratio = num_df.isnull().sum() / len(num_df)
    num_analysis_df['null_ratio'] = null_ratio

    return num_analysis_df

def calculate_statistics(df, num_cat):
    cat_df = df.iloc[:, :num_cat]
    num_df = df.iloc[:, num_cat:]

    cat_analysis_df = calculate_categorical_statistics(cat_df)
    num_analysis_df = calculate_numerical_statistics(num_df)

    return cat_analysis_df, num_analysis_df

def rename_columns(df, suffix):
    """Rename columns of the dataframe."""
    df.columns = [col + suffix for col in df.columns]
    return df

def compare_statistics(config, path, num_cat, real_cat, real_num):
    df = pd.read_csv(path)
    gen_cat, gen_num = calculate_statistics(df.copy(), num_cat)

    merged_cat = rename_columns(gen_cat, "_gen").join(rename_columns(real_cat, "_real"))
    merged_num = rename_columns(gen_num, "_gen").join(rename_columns(real_num, "_real"))

    format_df = lambda x: x.applymap(lambda y: '{:.4f}'.format(y))
    filename = path.split("/")[-1][:-4]
    
    format_df(merged_cat).to_csv(os.path.join(config["output_path"], "summary", f"{filename}_cat_summary.csv"), index=False)
    format_df(merged_num).to_csv(os.path.join(config["output_path"], "summary", f"{filename}_num_summary.csv"), index=False)
    return

def calculate_representative_value(path, num_cat, real_cat, real_num):
    """Calculate the score based on the differences between generated and real dataset summaries"""
    df = pd.read_csv(path)
    gen_cat, gen_num = calculate_statistics(df, num_cat=num_cat)

    diff_cat = (gen_cat - real_cat).abs().sum()
    diff_num = (gen_num - real_num).abs().sum()

    return {
        "setting": path.split("/")[-1],
        "most_frequent_ratio": diff_cat["most_frequent_ratio"],
        "least_frequent_ratio": diff_cat["least_frequent_ratio"],
        "cat_null_ratio": diff_cat["null_ratio"],
        "mean": diff_num["mean"],
        "std": diff_num["std"],
        "num_null_ratio": diff_num["null_ratio"]
        }


def get_seed_from_filename(filename):
    """
    Extract seed value from the filename.
    """
    for part in filename.split("_"):
        if part.isdigit() and int(part) in [0,1,2,3,4]:
            return int(part)
    return None

def analyze_stat_by_aki(real_data, synthetic_data, class_mapping, config):
    """
    Analyze statistics based on AKI values.
    """
    results = {}

    O_AKI_ratio = synthetic_data['O_AKI'].map(class_mapping).value_counts()[1] / len(synthetic_data)
    results.update({"aki_ratio": round(O_AKI_ratio, 4)})
    
    for aki_value in [0, 1]:
        filtered_real = real_data[real_data['O_AKI'].map(class_mapping) == aki_value]
        filtered_syn = synthetic_data[synthetic_data['O_AKI'].map(class_mapping) == aki_value]
        
        aki_stat = StatAnalysis(filtered_real, filtered_syn, config, significance_level=0.05, plot=False)
        stat_result = aki_stat.analyze(count=False, avg=True)
        
        results.update({
            f"numeric_t (aki={aki_value})": stat_result["numeric_t"],
            f"numeric_ks (aki={aki_value})": stat_result["numeric_ks"],
            f"categorical_chi2 (aki={aki_value})": stat_result["categorical_chi2"],
        })
    return results
 
def sample_data(data, scale, random_state):
    if scale == 1:
        return data.copy()
    else:
        num_rows_to_sample = scale
    return data.sample(n=num_rows_to_sample, random_state=random_state)


def inverse_boxcox_transform(df, study, input_path):
    if study == "BKC":
        zero_columns = ['B_ALT', 'Dur_before_Op', 'SPARK']
    else:
        zero_columns = {
            "SNUH": ['B_UPCR', 'B_hsCRP', 'Dur_before_Op', 'Op_EST_Dur', 'SPARK'],
            "SNUBH": ['B_Neutrophil', 'Dur_before_Op', 'SPARK'],
            "BRMH": ['B_ALT', 'Dur_before_Op', 'SPARK'],
            "KNUH": ['Dur_before_Op', 'SPARK'],
            "CAUH": ['Dur_before_Op'],
            "AMC": ['B_BIL', 'B_WBC', 'Dur_before_Op', 'Op_EST_Dur', 'SPARK'],
            "KUMC": ['B_Glucose', 'Dur_before_Op']
        }[study]
        
    lambdas= pd.read_csv(os.path.join(input_path, 'lambda_values.csv'))
    lambda_dict = dict(zip(lambdas["Column"], lambdas["Lambda"]))
    results = df.copy()
    
    for col in lambdas["Column"].values:
        non_na_data = df[col].dropna()
        lambda_val = lambda_dict[col]
        
        # Box-Cox Transformation                
        transformed_data = inv_boxcox(non_na_data, lambda_val)
        results.loc[non_na_data.index, col] = transformed_data
                
    for col in zero_columns:
        results[col] -= 1
        
    return results

def eval_pipeline(config, target_path, hosp2columns_for_pred, trg, seed):   
    input_path = config["input_path"].replace("240524", "240609")
    output_path = config["output_path"]
    num_cat = config["nc"]

    # Load config from JSON (new format) or fall back to pickle (old format)
    import json
    config_json_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, 'r') as f:
            info = json.load(f)
    else:
        # Fall back to pickle for backward compatibility
        info = pd.read_pickle(os.path.join(input_path, "info_dict.pickle"))
    
    class2raw = pd.read_pickle(os.path.join(input_path, "class2raw.pickle"))
    assert num_cat == info["cat_col_num"]
    
    # Ensure output directory exists
    summary_path = os.path.join(output_path, "summary")
    feat_importance_path = os.path.join(output_path, "feat_importance")
    feat_importance_file_path = os.path.join(feat_importance_path, os.path.basename(target_path))
    
    os.makedirs(summary_path, exist_ok=True)
    os.makedirs(feat_importance_path, exist_ok=True)

    # Load real dataset splits
    real = create_label_table(input_path, info, seed, dataset_type="trainvalid").copy()
    real_test = create_label_table(input_path, info, seed, dataset_type="test").copy()

    real_test = inverse_boxcox_transform(real_test, config["study"], input_path)
    real = inverse_boxcox_transform(real, config["study"], input_path)
    
    # Calculate statistics on real data
    real_cat, real_num = calculate_statistics(real.copy(), num_cat=num_cat)
    
    # Initialize results dataframe
    columns = [
        "setting", "most_frequent_ratio", "least_frequent_ratio" "cat_null_ratio", 
        "mean", "std", "num_null_ratio", "auroc_CB", "auprc_CB",
        'categorical_chi2', 'numeric_t', 'numeric_ks', 'null_columns',
        'categorical_chi2 (aki=0)','categorical_chi2 (aki=1)',
        'numeric_t (aki=0)','numeric_t (aki=1)',
        'numeric_ks (aki=0)','numeric_ks (aki=1)', 'aki_ratio', 'trg'
    ]
    results = pd.DataFrame(columns=columns)

    # Evaluation 1. TSTR
    # Load training and testing datasets
    X_train = pd.read_csv(target_path)
    X_train = inverse_boxcox_transform(X_train, config["study"], input_path)

    if len(real) < len(X_train):
        X_train = sample_data(X_train, len(real), seed)
    assert len(real) == len(X_train)
    
    # X_train = real.copy() ########################################################
    X_test = real_test.copy()

    X_train.iloc[:, :num_cat] = X_train.iloc[:, :num_cat].apply(lambda x: x.map(class2raw))
    X_test.iloc[:, :num_cat] = X_test.iloc[:, :num_cat].apply(lambda x: x.map(class2raw))

    # Extract outcome labels
    if X_train["O_AKI"].nunique() <= 1:
        return
    if X_train["O_AKI"].isnull().sum() > 1:
        return
    y_train = list(X_train["O_AKI"].values)
    y_test = list(X_test["O_AKI"].values)

    # Filter columns to exclude outcome columns for prediction using configuration mapping
    predictor_columns = hosp2columns_for_pred["total"]  # Column names used for prediction
    X_train = X_train[predictor_columns]
    X_test = X_test[predictor_columns]
    num_cat_for_pred = len(hosp2columns_for_pred["categorical"])
    
    model, _, auroc, auprc = evaluate_TSTR(X_train.copy(), X_test, y_train, y_test, num_cat_for_pred, seed, extract_feat_importance=feat_importance_file_path)
    
    # Evaluation 2: Statistics comparison
    result = calculate_representative_value(target_path, num_cat, real_cat.copy(), real_num.copy())
    compare_statistics(config, target_path, num_cat, real_cat.copy(), real_num.copy())
    
    # Evaluation 3: Statistical tests
    gen = pd.read_csv(target_path) # Load again!
    stat = StatAnalysis(real.copy(), gen.copy(), config, significance_level=0.05, plot=False)
    result.update(stat.analyze(count=False, avg=True))

    results_by_aki = analyze_stat_by_aki(real.copy(), gen.copy(), class2raw, config)
    result.update(results_by_aki)

    result["trg"] = trg
    result["auroc_CB"], result["auprc_CB"] = auroc, auprc

    results = pd.DataFrame([result])
    results.to_csv(os.path.join(summary_path, target_path.split("/")[-1]), index=False)
    print(os.path.join(summary_path, target_path.split("/")[-1]))
    return auroc, auprc, sum(y_train) / len(y_train)

if __name__ == "__main__":
    import argparse
    from mmtg.config import ex, add_hospital_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=str, default='SNUBH', help='Study name')
    args = parser.parse_args()
    
    study = args.study
    
    # Load config using Sacred system
    @ex.automain
    def main(_config):
        return _config
    
    # Get Sacred config
    temp_config = {'study': study}
    temp_config = add_hospital_config(temp_config, "eval_pipeline", None)
    config = main()
    
    # Extract values from config
    n_w = config.get('num_loss_weight', 0.1)
    d = config.get('dropout', 0.1)
    temp = config.get('temperature', 1.0)
    null_sample = config.get('null_sample', True)
    unmask_by = config.get('unmask_by', 'random')
    postpro = "constaki"  # Can be moved to config if needed
    sched = config.get('mask_scheduler', 'pow1')
    niter = config.get('n_iter', 20)
    
    hosp2columns_for_pred = extract_columns_for_pred(data_base_path=os.path.dirname(os.path.dirname(config['input_path'])))[study][study]

    auroc_list = []
    auprc_list = []
    aki_list = []
    for seed in [0,1,2]:
        for trg in [study]:
            if study == trg:
                filename = f"table_MLM_{sched}_{n_w}_mini_abs_False_False_5e-05_{d}_{seed}_{niter}_categorical_{unmask_by}_{null_sample}_{temp}_trainvalid_gen_{postpro}.csv"
            
            # Use output_path from config
            root = config['output_path']
            target_root = glob.glob(os.path.join(root, filename))
            
            for target_path in sorted(target_root):
                auroc, auprc, aki = eval_pipeline(config, target_path, hosp2columns_for_pred, trg, seed)
                
                auroc_list.append(auroc)
                auprc_list.append(auprc)
                aki_list.append(aki)
        print(study)
        print(round(np.mean(auroc_list),4), round(np.mean(auprc_list),4), round(np.mean(aki_list), 4))
        print("auroc_list", auroc_list)
        print("auprc_list", auprc_list)
        print("aki_list", aki_list)
