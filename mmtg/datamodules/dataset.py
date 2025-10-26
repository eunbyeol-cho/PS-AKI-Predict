import os
import random
import torch
import math
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from ..modules.mask_schedule import schedule


class MMTGDataset(torch.utils.data.Dataset):
    def __init__(self, _config, split):
        super().__init__()

        self.split = split 
        self.seed = _config['seed']
        self.nc, self.nn, self.ntext = _config['nc'], _config['nn'], _config['ntext']
        self.ntable = self.nc + self.nn
        self.use_text = len(_config['modality']) != 1
        self.cat_null_id = _config['cat_null_id']
        self.num_null_id = _config['num_null_id']
        self.spark_features = _config["spark_features"]
    
        if _config['target_study']:
            data_root = _config['input_path'].replace(_config["study"], _config["target_study"])
            conditioned_data, conditioned_null_type_data, self.spark_features_indices = process_data(_config['input_path'], data_root, self.spark_features)
            self.fold = os.path.join(data_root, f'fold/snuh_{self.seed}_fold_split.csv')  
            hit_idcs = self._get_fold_indices()
            
            self.input_data = conditioned_data[hit_idcs]
            self.null_type_data = conditioned_null_type_data[hit_idcs]
            # print(self.input_data[:, self.spark_features_indices[:5]])
            # print(self.input_data[:, self.spark_features_indices[5:]])

        else:
            data_root = _config['input_path']
            self.fold = os.path.join(data_root, f'fold/snuh_{self.seed}_fold_split.csv')  
            hit_idcs = self._get_fold_indices()

            self.input_data = np.load(os.path.join(data_root, 'input_ids.npy'))[hit_idcs]
            self.null_type_data = np.load(os.path.join(data_root, 'null_type_ids.npy'))[hit_idcs]
        
        self.data_dict = self._get_data()
        self.test_only = _config['test_only']
        self.modeling = _config['modeling']
        assert self.modeling == 'MLM'
        self._set_mlm_params(_config, data_root)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        mlm_prob = self._get_mlm_probability()
        cat_data = self._prepare_data('cat', index, mlm_prob)
        num_data = self._prepare_data('num', index, mlm_prob)

        # Include textual data if required
        if self.use_text:
            text_data = self._prepare_text(index)
            return {'cat': cat_data, 'num': num_data, 'text': text_data}
        else:
            return {'cat': cat_data, 'num': num_data}

    def collator(self, samples):
        input, out = {}, {}

        input_types = ['cat', 'num', 'text'] if self.use_text else ['cat', 'num']

        # Collate samples
        for input_type in input_types:
            input = self._collate_samples(input, samples, input_type)

        out["net_input"] = input
        return out

    def _set_mlm_params(self, _config, data_root):
        self.mask_scheduler = _config['mask_scheduler']
        self.cat_null_token_id = _config['cat_null_id']
        self.num_null_token_id = _config['num_null_id']
        self.mask_token_id = self.cat_null_token_id + 1
        self.test_mlm_probability = _config['test_mlm_probability']

        #Test options
        self.fixed_test_mask = (_config['fixed_test_mask']) & self.test_only        
        if self.test_only:
            print(f"Mask: {self.test_mlm_probability}. Scheduler {self.mask_scheduler}. Conditioned on SPARK: {self.fixed_test_mask}")
        else:
            print(f"Mask Scheduler: {self.mask_scheduler}. Conditioned on SPARK: {self.fixed_test_mask}")

    def _get_mlm_probability(self):
        if self.test_only:
            return self.test_mlm_probability
        elif isinstance(self.mask_scheduler, float):
            return self.mask_scheduler
        t = torch.zeros(1).uniform_(0, 1).item()
        return schedule(t, total_unknown=self.ntable, method=self.mask_scheduler)

    def _prepare_data(self, data_type, index, mlm_prob):
        if data_type == 'cat':
            input = torch.LongTensor(self.data_dict[data_type]['input'][index])
        elif data_type == 'num':
            input = torch.FloatTensor(self.data_dict[data_type]['input'][index])

        null_type = torch.LongTensor(self.data_dict[data_type]['null_type'][index])
        token_type = torch.LongTensor(self.data_dict[data_type]['token_type'][index])
                
        fixed_mask_indices = self._get_fixed_mask_indices(data_type, input)
        input_ids, input_labels, unk_mask = self.mask_tokens(input, mlm_prob, fixed_mask_indices=fixed_mask_indices)
        null_ids = torch.LongTensor(np.where(unk_mask, 2, null_type))
        
        return {
            'input_ids':input_ids,
            'input_labels':input_labels,
            'null_ids':null_ids,
            'null_labels':null_type,
            'token_type':token_type,
        }
    
    def _prepare_text(self, index):
        input_ids = torch.LongTensor(self.data_dict['text']['input'][index])
        token_type = torch.LongTensor(self.data_dict['text']['token_type'][index])
        
        return {
            'input_ids':input_ids,
            'token_type':token_type,
        }

    def _default_values(self, input, null_type):
        return input.clone(), input.clone(), null_type.clone()

    def _get_fixed_mask_indices(self, data_type, input):
        if not self.fixed_test_mask:
            return None

        # Initialize a mask with all True (fixed by default)
        fixed_mask_indices = np.ones_like(input, dtype=bool)

        # Determine indices based on data type
        if data_type == 'cat':
            feature_indices = [index for index in self.spark_features_indices if index < self.nc]
        elif data_type == 'num':
            feature_indices = [index - self.nc for index in self.spark_features_indices if index >= self.nc]
        else:
            raise ValueError("Invalid data type specified. Use 'cat' or 'num'.")
        
        # Set selected features to False (not fixed)
        fixed_mask_indices[feature_indices] = False
        return fixed_mask_indices

    def _collate_samples(self, input, samples, input_type):
        input[f"{input_type}_input_ids"] = torch.stack([s[input_type]["input_ids"] for s in samples])
        input[f"{input_type}_token_type"] = torch.stack([s[input_type]["token_type"] for s in samples])
        if input_type != 'text':
            # for key in ["input", "input_labels", "null_ids", "null_labels"]:
            for key in ["input_labels", "null_ids", "null_labels"]:
                input[f"{input_type}_{key}"] = torch.stack([s[input_type][key] for s in samples])
        return input
    
    def _get_data(self): 
        # Define data types and their indices
        data_types = {
            'cat': (0, self.nc),
            'num': (self.nc, self.ntable)
        }

        # Include 'text' data type if required
        if self.use_text:
            data_types['text'] = (self.ntable, None)

        data = {}

        for data_type, (start_idx, end_idx) in data_types.items():
            # Slice input data according to start and end indices
            input_data = self.input_data[:, start_idx:end_idx]
            
            # Prepare null_type data. Only for 'cat' and 'num' data types.
            null_type = self.null_type_data[:, start_idx:end_idx] if data_type != 'text' else None
            
            # If data type is 'text', token type is all ones. Otherwise, it's all zeros.
            token_type = np.ones_like(input_data) if data_type == 'text' else np.zeros_like(input_data)

            data[data_type] = {
                'input': input_data,
                'null_type': null_type,
                'token_type': token_type
            }
            
        return data

    def _get_fold_indices(self):
        split_idx = {'train':1, 'valid':2, 'test':0}
        splits = pd.read_csv(self.fold)['fold'].values
        if self.split == "trainvalid":
            idcs = np.where(splits != 0)[0]
        else:
            hit = split_idx[self.split]
            idcs = np.where(splits == hit)[0]
        return idcs

    def mask_tokens(self, inputs, mlm_prob, special_tokens_masks=None, fixed_mask_indices=None):
        """
        Ref: https://huggingface.co/transformers/v4.9.2/_modules/transformers/data/data_collator.html
        """
        inputs = inputs.unsqueeze(0)
        labels = inputs.clone()
        init_no_mask = torch.full_like(inputs, False, dtype=torch.bool)

        if fixed_mask_indices is not None:
            masked_indices = torch.unsqueeze(torch.BoolTensor(fixed_mask_indices), dim=0)
        
        else:
            if special_tokens_masks is not None:
                special_tokens_mask = reduce(lambda acc, el: acc | (inputs == el), special_tokens_masks, init_no_mask)
                probability_matrix = torch.full(labels.shape, mlm_prob)
                special_tokens_mask = special_tokens_mask.bool()
                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            else : 
                special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
                probability_matrix = torch.full(labels.shape, mlm_prob)
            masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.mask_token_id
        return inputs.squeeze(0), labels.squeeze(0), masked_indices.squeeze(0)


def mmtg_data_loader(_config):
    data_loaders = dict()
    for split in ['train', 'valid', 'test', "trainvalid"]:
        dataset = MMTGDataset(_config, split)
        shuffle = True if split == 'train' else False
        if _config["test_only"]:
            shuffle = False
        data_loaders[split] = DataLoader(
            # dataset, collate_fn=dataset.collator, batch_size=_config['per_gpu_batchsize'], num_workers=8, shuffle=shuffle
            dataset, collate_fn=dataset.collator, batch_size=_config['per_gpu_batchsize'], num_workers=0, shuffle=shuffle
            )
    return data_loaders


def load_data(data_path):
    import json
    # Load config from JSON (new format) or fall back to pickle (old format)
    config_json_path = os.path.join(data_path, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, 'r') as f:
            info = json.load(f)
    else:
        # Fall back to pickle for backward compatibility
        info = pd.read_pickle(os.path.join(data_path, "info_dict.pickle"))
    
    return {
        'info': info,
        'class2raw': pd.read_pickle(os.path.join(data_path, "class2raw.pickle")),
        'znorm': pd.read_csv(os.path.join(data_path, "znorm.csv")),
        'input_data': np.load(os.path.join(data_path, "unnormalized_input_ids.npy")),
        'null_type_data': np.load(os.path.join(data_path, 'null_type_ids.npy')) if 'null_type_ids.npy' in os.listdir(data_path) else None
    }

def normalize_numeric(data, mean, std, is_null):
    return ((data - mean) / std) * (~is_null)

def map_categorical(data, mapper, default_value):
    return data.map(lambda x: mapper.get(x, default_value))

def verify_cat_consistency(before, after, before_null_id, after_null_id):
    """
    There can be cases where mapping fails due to the type of the keys being strings/integers during the double mapping process of `raw_to_class`.
    Therefore, it would be necessary to place a code that checks whether the proportion of null values remains the same before and after the mapping.
    """
    
    # Convert numpy arrays to pandas Series
    before_series = pd.Series(before)
    after_series = pd.Series(after)

    # Get value counts
    before_counts = before_series.value_counts()
    after_counts = after_series.value_counts()
    
    before_counts_set = set(before_counts.values)
    after_counts_set = set(after_counts.values)

    # Print counts for comparison
    # print("Before value counts:\n", before_counts)
    # print("After value counts:\n", after_counts)

    # Check if the sets of counts are equivalent
    assert before_counts_set == after_counts_set, "The distributions of values have changed."
    
    # Count occurrences of the null identifiers
    before_count = np.sum(before == before_null_id)
    after_count = np.sum(after == after_null_id)

    # Print the counts
    # print(f'Before null count: {before_count}')
    # print(f'After null count: {after_count}')

    # Ensure the counts of specific null identifiers remain the same
    assert before_count == after_count, "Null value counts changed after mapping."

def verify_num_consistency(num_null_loc, after, after_null_id):
    
    # Count occurrences of the null identifiers
    # before_count = np.sum(before == before_null_id)
    before_count = num_null_loc.sum()
    after_count = np.sum(after == after_null_id)

    # Print the counts
    # print(f'Before null count: {before_count}')
    # print(f'After null count: {after_count}')

    # Ensure the counts of specific null identifiers remain the same
    assert before_count == after_count, "Null value counts changed after mapping."

def process_data(input_path, target_input_path, spark_features):
    source = load_data(input_path)
    target = load_data(target_input_path)

    conditioned_data = np.zeros((target['input_data'].shape[0], source['input_data'].shape[1]))
    conditioned_data[:, :source['info']["cat_col_num"]] = source['info']["cat_null_ids"]
    conditioned_data[:, source['info']["cat_col_num"]:] = source['info']["num_null_ids"]

    conditioned_null_type_data = np.ones((target['input_data'].shape[0], source['input_data'].shape[1]))
    conditioned_null_type_data = conditioned_null_type_data.astype(bool)
    spark_features_indices = []
    for feat in spark_features:

        if feat in target['info']["column_order"]:

            target_feat_idx = target['info']["column_order"].index(feat)
            source_feat_idx = source['info']["column_order"].index(feat)
            spark_features_indices.append(source_feat_idx)
            
            if feat in ["A_Sex", "Op_Dep", "Op_Type", "AB_DM", "D_RASB_90"]:
                temp = map_categorical(pd.Series(target['input_data'][:, target_feat_idx]), target['class2raw'], -1)
                
                source_mapped_values = np.unique(source['input_data'][:, source_feat_idx])
                raw_to_class = {v: k for k, v in source['class2raw'].items() if k in source_mapped_values}
                raw_to_class[-1] = source['info']["cat_null_ids"]
                
                mapped_data = temp.map(raw_to_class)
                conditioned_data[:, source_feat_idx] = mapped_data
                conditioned_null_type_data[:, source_feat_idx] = target['null_type_data'][:, target_feat_idx]
                
                verify_cat_consistency(target['input_data'][:, target_feat_idx], mapped_data, target['info']["cat_null_ids"], source['info']["cat_null_ids"])
                

            elif feat in ["A_Age", "B_Cr_near", "Op_EST_Dur"]:
                feat_mean = source['znorm']['mean'].values[source_feat_idx - source['info']["cat_col_num"]]
                feat_std = source['znorm']['std'].values[source_feat_idx - source['info']["cat_col_num"]]
                num_null_loc = target['null_type_data'][:, target_feat_idx]

                conditioned_data[:, source_feat_idx] = normalize_numeric(target['input_data'][:, target_feat_idx], feat_mean, feat_std, num_null_loc)
                conditioned_null_type_data[:, source_feat_idx] = target['null_type_data'][:, target_feat_idx]
                
                verify_num_consistency(num_null_loc, conditioned_data[:, source_feat_idx], source['info']["num_null_ids"]) 

            else:
                assert False, "Unexpected feature"
    
    return conditioned_data, conditioned_null_type_data, spark_features_indices