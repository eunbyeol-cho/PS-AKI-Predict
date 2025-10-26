import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import scipy.stats
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

from mmtg.utils.eval_pipeline import create_label_table

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

def load_real_data(input_path, seed):
    # input_path should be the full path to the znorm directory for a specific study
    class2raw = pd.read_pickle(os.path.join(input_path, "class2raw.pickle"))
    
    # Load config from JSON (new format) or fall back to pickle (old format)
    import json
    config_json_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, 'r') as f:
            info = json.load(f)
    else:
        # Fall back to pickle for backward compatibility
        info = pd.read_pickle(os.path.join(input_path, "info_dict.pickle"))
    
    hosp2columns_for_pred = pd.read_pickle('compatible_columns.pickle')[hosp]

    real_train = create_label_table(input_path, info, seed, dataset_type="trainvalid")
    real_test = create_label_table(input_path, info, seed, dataset_type="test")

    real_train = map_categorical_data(real_train, info, class2raw)
    real_test = map_categorical_data(real_test, info, class2raw)

    return class2raw, info, hosp2columns_for_pred, real_train, real_test
    
def map_categorical_data(data, info, class2raw):
    data.iloc[:, :info["cat_col_num"]] = data.iloc[:, :info["cat_col_num"]].apply(lambda x: x.map(class2raw))
    return data

def sample_balanced_data(data, scale, random_state, value_counts=None):
    if value_counts is None:
        # Simple scaling without balancing
        if scale == 1:
            return data.copy()
        num_rows_to_sample = round(len(data) * scale) if (0 < scale < 1) else round(scale)
        return data.sample(n=num_rows_to_sample, random_state=random_state, replace=False)

    else:
        # Balancing based on target value_counts
        sampled_data = []
        all_sampled_indices = set()  # To keep track of sampled indices

        for value_combination, ratio in value_counts.items():
            # Extract rows matching the current value_combination
            matching_rows = data[(data["O_AKI"] == value_combination[0]) &
                                 (data["Op_Dep"] == value_combination[1])]

            if len(matching_rows) == 0:
                raise ValueError(f"No matching rows found for value_combination: {value_combination}")

            # Sample the rows
            sample_size = round(ratio * scale)
            sampled_subset = matching_rows.sample(
                n=min(sample_size, len(matching_rows)), random_state=random_state, replace=False
            )

            sampled_data.append(sampled_subset)
            all_sampled_indices.update(sampled_subset.index)

        # Combine all sampled subsets into one DataFrame
        balanced_sample = pd.concat(sampled_data, axis=0)

        # If there are still missing samples to reach the scale, sample from unsampled data
        current_size = len(balanced_sample)
        if current_size < scale:
            unsampled_data = data.drop(index=all_sampled_indices)
            additional_rows = unsampled_data.sample(
                n=scale - current_size, random_state=random_state, replace=False
            )
            balanced_sample = pd.concat([balanced_sample, additional_rows], axis=0)

        return balanced_sample.reset_index(drop=True)



def sample_data(data, scale, random_state, aki_ratio=None):
    if aki_ratio is None:
        if scale == 1:
            return data.copy()
        num_rows_to_sample = int(len(data) * scale) if (0 < scale < 1) else scale
        return data.sample(n=num_rows_to_sample, random_state=random_state)
    else:
        # Ensure AKI ratio in the sampled data
        if not (0 <= aki_ratio <= 1):
            raise ValueError("aki_ratio must be between 0 and 1.")
        
        data_0 = data[data['O_AKI'] == 0]
        data_1 = data[data['O_AKI'] == 1]
        
        if 0 < scale < 1:
            num_rows_to_sample = int(len(data) * scale)
        else:
            num_rows_to_sample = scale  # Use scale directly if >1
        
        # Calculate the number of samples for each class based on the desired ratio
        num_samples_1 = int(num_rows_to_sample * aki_ratio)
        num_samples_0 = num_rows_to_sample - num_samples_1
        
        if len(data_1) < num_samples_1:
            raise ValueError(f"Not enough O_AKI=1 samples available for the requested aki_ratio={aki_ratio}.")
        
        sampled_data_0 = data_0.sample(n=num_samples_0, random_state=random_state, replace=False)
        sampled_data_1 = data_1.sample(n=num_samples_1, random_state=random_state, replace=False)

        # Combine and shuffle the sampled data
        return pd.concat([sampled_data_0, sampled_data_1])

def bootstrap_delong_test(ground_truth, predictions_one, predictions_two, n_iterations=1000, seed=0):
    p_value = delong_roc_test(ground_truth, predictions_one, predictions_two)
    return (10 ** p_value).item()

def bootstrap_wilcoxon_test(ground_truth, predictions_one, predictions_two, n_iterations=1000, seed=0):
    """Perform bootstrap sampling and apply the Wilcoxon test to compute p-values and confidence intervals."""
    np.random.seed(seed)
    auroc_scores_diff = []
    n = len(ground_truth)

    for _ in range(n_iterations):
        indices = np.random.randint(0, n, n)
        auroc_one = roc_auc_score(ground_truth[indices], predictions_one[indices])
        auroc_two = roc_auc_score(ground_truth[indices], predictions_two[indices])
        auroc_scores_diff.append(auroc_one - auroc_two)

    stat, p_value = wilcoxon(auroc_scores_diff)
    confidence_interval = np.percentile(auroc_scores_diff, [2.5, 97.5])
    return p_value, f"({round(confidence_interval[0], 3)}, {round(confidence_interval[1], 3)})"

def evaluate_models_and_collect_results(seed_iter, model_types, X_train_list, y_train_list, X_test, y_test, cat_cols, seed, src_org, trg_org, src_scale, trg_scale, results_df):
    """Trains models and collects performance metrics."""
    pred_prob = {}
    for model_label, X_train, y_train in zip(model_types, X_train_list, y_train_list):
        model, y_pred_prob, auroc, auprc = evaluate_TSTR(X_train.copy(), X_test, y_train, y_test, len(cat_cols), seed)
        
        results_df = results_df.append({
            'src_org': src_org, 'trg_org': trg_org, 'src_scale': src_scale, 'trg_scale': trg_scale, 'seed': seed,
            'model_type': model_label, 'auroc': auroc, 'auprc': auprc, 
            'num_train': len(X_train), 'iter': seed_iter, 'aki1_train':sum(y_train)
        }, ignore_index=True)
        print(f"{model_label}: AUROC={auroc}, AUPRC={auprc}")
        pred_prob[model_label] = y_pred_prob
    return results_df, pred_prob

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_org', type=str, default='AMC', 
                        help='Source organization (study)')
    parser.add_argument('--src_input', type=str, 
                        help='Input path for source data znorm directory')
    parser.add_argument('--src_output', type=str, 
                        help='Output path for source synthetic data')
    args = parser.parse_args()
    
    src_org = args.src_org
    src_input_root = args.src_input
    src_output_root = args.src_output
    
    src_orgs = [src_org]
    
    src_scales = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    trg_scales = [250, 500, 1000, 2000]
    seed_iterations_value = 30
    
    seeds = [0,1,2]

    postpro = "em"
    dropout=0.1
    aki_ratio=None
    # aki_ratio=0.11

    hyperparams_list = {
        'SNUH': [(20, "pow1", "random", postpro, dropout)],
        'SNUBH': [(20, "pow1", "random", postpro, dropout)],
        'AMC': [(20, "pow1", "random", postpro, dropout)],
    }
    
    # Build path mapping for each study
    all_studies = ['SNUH', 'SNUBH', 'BRMH', 'KNUH', 'CAUH', 'AMC', 'KUMC', 'BKC']
    study_input_paths = {}
    for study in all_studies:
        if study == src_org:
            study_input_paths[study] = src_input_root
    
    for src_org in src_orgs:
        trg_orgs = ['SNUH', 'SNUBH', 'BRMH', 'KNUH', 'CAUH', 'AMC']
        if src_org in trg_orgs:
            trg_orgs.remove(src_org)
        syn_root = src_output_root
        
        for hyperparams in hyperparams_list[src_org]:
            results_df = pd.DataFrame()
            niter, sched, unmask_by, postpro, dropout = hyperparams
            
            for trg_org in trg_orgs:  

                for trg_scale in trg_scales:
                    for src_scale in src_scales: 
                        if (src_scale < trg_scale) and (trg_scale != 0):
                            continue
                        ground_truth, predictions_trg_only, predictions_trg_real, predictions_trg_syn = [], [], [], []
                        
                        for seed in seeds:  
                            scale_factor = 1 if trg_scale == 0 else trg_scale
                            seed_iterations = 1 if trg_scale == 0 else seed_iterations_value 

                            for si in range(seed_iterations):
                                print(f"\nProcessing: Source={src_org}, Source Scale={src_scale}, Target={trg_org}, Target Scale={trg_scale}, Seed={seed}, Iter={si}, Postpro={postpro}, aki_ratio={aki_ratio}")
                                _, _, _, trg_train, trg_test = load_real_data(study_input_paths[trg_org], seed)
                                try:
                                    trg_train_subset = sample_data(trg_train, scale_factor, si)
                                    error_occurred = False
                                except ValueError as e:
                                    print("Sampling error: Attempted to sample more than available without replacement. Skipping...")
                                    error_occurred = True
                                    continue  # Break out of the inner loop

                                # Load source real data for training
                                src_class2raw, src_info, common_columns_for_pred, src_real_train, src_real_test = load_real_data(study_input_paths[src_org], seed)

                                # Load sources ynthetic data for training
                                if postpro == "constaki":
                                    filename = f"table_MLM_{sched}_0.1_mini_abs_False_False_5e-05_{dropout}_{seed}_{niter}_categorical_{unmask_by}_True_1_trainvalid_gen_{postpro}.csv"
                                else:
                                    filename = f"table_MLM_{sched}_0.1_mini_abs_False_False_5e-05_{dropout}_{seed}_{niter}_categorical_{unmask_by}_True_1_trainvalid_{trg_org}_gen_{postpro}.csv"
                                
                                src_syn_train = pd.read_csv(os.path.join(syn_root, filename))
                                src_syn_train = map_categorical_data(src_syn_train, src_info, src_class2raw)

                                
                                try:
                                    trg_value_counts = trg_train[["O_AKI", "Op_Dep"]].value_counts() / len(trg_train)
                                    src_real_train_subset = sample_balanced_data(src_real_train, src_scale, si, trg_value_counts).copy()
                                    src_syn_train_subset = sample_balanced_data(src_syn_train, src_scale, si, trg_value_counts).copy()
                                    error_occurred = False

                                except ValueError as e:
                                    print("Sampling error: Attempted to sample more than available without replacement. Skipping...")
                                    error_occurred = True
                                    continue  # Break out of the inner loop

                                # Concatenate training sets
                                src_input_path = os.path.join(input_root, f'20240609/{src_org}', 'znorm')
                                trg_input_path = os.path.join(input_root, f'20240609/{trg_org}', 'znorm')

                                trg_train_subset = inverse_boxcox_transform(trg_train_subset.copy(), trg_org, trg_input_path)
                                src_real_train_subset = inverse_boxcox_transform(src_real_train_subset, src_org, src_input_path)
                                src_syn_train_subset = inverse_boxcox_transform(src_syn_train_subset, src_org, src_input_path)
                                X_test = inverse_boxcox_transform(trg_test.copy(), trg_org, trg_input_path)
                                print(f"Shapes - trg_train_subset: {trg_train_subset.shape}, src_real_train_subset: {src_real_train_subset.shape}, src_syn_train_subset: {src_syn_train_subset.shape}, X_test: {X_test.shape}")
                                
                                X_train_trg_only = trg_train_subset
                                X_train_trg_aug_w_src_real = pd.concat([src_real_train_subset.copy(), trg_train_subset], ignore_index=True)
                                X_train_trg_aug_w_src_syn = pd.concat([src_syn_train_subset.copy(), trg_train_subset], ignore_index=True)
                                # X_test = trg_test.copy()

                                # Extract labels
                                y_train_trg_only = list(X_train_trg_only["O_AKI"].values)
                                y_train_trg_aug_w_src_real = list(X_train_trg_aug_w_src_real["O_AKI"].values)
                                y_train_trg_aug_w_src_syn = list(X_train_trg_aug_w_src_syn["O_AKI"].values)
                                y_test = list(trg_test["O_AKI"].values)
                                
                                cat_cols = common_columns_for_pred[trg_org]["categorical"]
                                num_cols = common_columns_for_pred[trg_org]["numerical"]

                                # Filter columns
                                X_train_trg_only = X_train_trg_only[cat_cols+num_cols]
                                X_train_trg_aug_w_src_real = X_train_trg_aug_w_src_real[cat_cols+num_cols]
                                X_train_trg_aug_w_src_syn = X_train_trg_aug_w_src_syn[cat_cols+num_cols]
                                X_test = X_test[cat_cols+num_cols]
                    
                                # Model evaluations
                                if src_org == trg_org:
                                    model_types = ['trg_only', 'trg_aug_w_src_syn']
                                    X_train_list = [X_train_trg_only, X_train_trg_aug_w_src_syn]
                                    y_train_list = [y_train_trg_only, y_train_trg_aug_w_src_syn]
                                else:
                                    model_types = ['trg_only', 'trg_aug_w_src_real', 'trg_aug_w_src_syn']
                                    if trg_scale == 0:
                                        X_train_list = [X_train_trg_only, src_real_train_subset[cat_cols + num_cols], src_syn_train_subset[cat_cols + num_cols]]
                                        y_train_list = [y_train_trg_only, list(src_real_train_subset["O_AKI"].values), list(src_syn_train_subset["O_AKI"].values)]
                                    else:
                                        X_train_list = [X_train_trg_only, X_train_trg_aug_w_src_real, X_train_trg_aug_w_src_syn]
                                        y_train_list = [y_train_trg_only, y_train_trg_aug_w_src_real, y_train_trg_aug_w_src_syn]

                                # Evaluate models and append results
                                results_df, pred_prob = evaluate_models_and_collect_results(
                                    si, model_types, X_train_list, y_train_list, X_test, y_test, cat_cols, seed, src_org, trg_org, src_scale, trg_scale, results_df
                                )
                                
                                # Now extract predictions for DeLong test
                                ground_truth.append(np.array(y_test)) # Convert list of labels to numpy array for indexing
                                predictions_trg_only.append(pred_prob['trg_only'])
                                predictions_trg_real.append(pred_prob['trg_aug_w_src_real'])
                                predictions_trg_syn.append(pred_prob['trg_aug_w_src_syn'])

                results_df_filename = f"results/delong_test/results_{postpro}_{src_org}_{sched}_{niter}_{unmask_by}_{dropout}_{seed_iterations_value}_{len(seeds)}_Opdep_Oaki_noBootstrap.csv"            
                results_df.to_csv(results_df_filename, index=False)