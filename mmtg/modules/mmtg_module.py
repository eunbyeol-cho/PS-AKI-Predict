import os
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from einops import rearrange
from .input2emb import MMTGInput2Emb
from .emb2out import MMTGEmb2Out
from .out2table import MMTGOut2Table
from .mask_schedule import schedule

class MMTGTransformer(nn.Module):
    def __init__(self, _config):
        super().__init__()

        self.test_only = _config['test_only']
        self.nc = _config['nc']
        self.nn = _config['nn']
        
        # Load config from JSON (new format) or fall back to pickle (old format)
        import json
        config_json_path = os.path.join(_config['input_path'], "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                self._info_dict = json.load(f)
        else:
            # Fall back to pickle for backward compatibility
            self._info_dict = pd.read_pickle(os.path.join(_config['input_path'], "info_dict.pickle"))
        
        self.aki_index = self._info_dict["column_order"].index("O_AKI") #62
        self.ntable = self.nc + self.nn
        self.cat_null_token_id = _config['cat_null_id']
        self.num_null_token_id = _config['num_null_id']
        self.mask_token_id = self.cat_null_token_id + 1
        self.mask_scheduler = _config['mask_scheduler']
        self.use_text = 'text' in _config['modality'] 
        self.n_iter = _config["n_iter"]
        self.temperature = _config['temperature']
        self.topk_filter_thres = _config['topk_filter_thres']
        self.sigmoid_temperature = _config['sigmoid_temperature']
        self.sample_by = _config['sample_by']
        self.unmask_by = _config['unmask_by']
        self.null_sample = _config['null_sample']
        self.fixed_test_mask = (_config['fixed_test_mask']) & self.test_only
        self.boxcox_transformation = _config['boxcox_transformation']

        self.spark_features_indices = []
        for feat in _config["spark_features"]:
            if _config["target_study"] != "KUMC" and feat != "Op_EST_Dur":            
                source_feat_idx = _config["column_order"].index(feat)
                self.spark_features_indices.append(source_feat_idx)

        if "classimbalance" in self.unmask_by :
            column_order = self._info_dict["column_order"]

            entropy_order = pd.read_csv(os.path.join(_config["input_path"], "class_imbalance_cat.csv"))["entropy"].values
            skewness_order = pd.read_csv(os.path.join(_config["input_path"], "class_imbalance_num.csv"))["column"].values

            self.cat_column_order = [column_order[:self.nc].index(col) for col in entropy_order]
            self.num_column_order = [column_order[self.nc:].index(col) for col in skewness_order]


        bert_config = {
            "tiny": (2, 128, 2),
            "mini": (4, 256, 4),
            "small": (4, 512, 8),
            "medium": (8, 512, 8),
            "base": (12, 768, 12),
        }

        num_layers, embed_dim, n_head = bert_config[_config['bert_type']]
        encoder_layer = TransformerEncoderLayer(
            dropout=_config["dropout"],
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=4*embed_dim,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True
        )
        
        self.bert = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.input2emb = MMTGInput2Emb(_config)
        self.emb2out = MMTGEmb2Out(_config)
        self.out2table = MMTGOut2Table(_config) if self.test_only else None

        cat_nunique = _config['cat_columns_numclass_list']
        self.softmax_mask = torch.full((_config['nc'], _config['cat_vocab_size']), -100000)#.to('cuda')
        cumulated = 0
        for s, v in enumerate(cat_nunique):
            self.softmax_mask[s, cumulated:cumulated+v] = 0
            cumulated += v
        self.softmax_layer = nn.LogSoftmax(dim=-1)
        self.gen_table = []

    def forward(self, **kwargs):
        x = self.input2emb(**kwargs)
        x = self.bert(x)
        x = self.emb2out(x, **kwargs)

        with torch.no_grad():
            self.out2table(x, **kwargs) if self.test_only else None
        return x

    @torch.no_grad()
    def iterative_decode(self, **kwargs):
        kwargs = self.move_to_cuda(kwargs)
        cat_steps, num_steps = map(self.calculate_steps, [kwargs['cat_input_labels'], kwargs['num_input_labels']])
        starting_temperature = self.temperature
    
        init_cond_cat = kwargs['cat_input_ids'].clone()
        init_cond_num = kwargs['num_input_ids'].clone()
        
        assert len(cat_steps) == len(num_steps)
        
        for iter, steps_until_x0 in zip(range(self.n_iter-1), reversed(range(self.n_iter-1))):
            self.temperature = self.update_temperature(starting_temperature, steps_until_x0)

            x = self.input2emb(**kwargs)

            x = self.bert(x)
            output = self.emb2out(x, **kwargs)

            cat_pred, num_pred, null_pred = self.generate_predictions(output)
            kwargs = self.update_predictions(kwargs, output, cat_pred, num_pred, null_pred, cat_steps[iter], num_steps[iter])
        
        if self.fixed_test_mask:
            spark_cat_indices = self.spark_features_indices[:5]
            spark_num_indices = [i-self.nc for i in self.spark_features_indices[5:]]
            assert (init_cond_cat[:, spark_cat_indices] != (kwargs['cat_input_ids'][:, spark_cat_indices])).sum().item() == 0
            assert (init_cond_num[:, spark_num_indices] != (kwargs['num_input_ids'][:, spark_num_indices])).sum().item() == 0
        
        finalized_output = self.finalize_output(kwargs)
        self.gen_table.append(finalized_output)
        return output, finalized_output

    def move_to_cuda(self, kwargs):
        device = torch.device("cuda")
        return {key: value.cuda() for key, value in kwargs.items()}

    def calculate_steps(self, input_labels):           
        num_token_masked = []
        column_num = input_labels.shape[1]

        for ratio in torch.linspace(0, 1, self.n_iter):
            mask_ratio = schedule(ratio, total_unknown=self.ntable, method=self.mask_scheduler)
            num_token_masked.append(int((mask_ratio * column_num).item()))
           
        steps = [num_token_masked[i] - num_token_masked[i+1] for i in range(self.n_iter-1)] 
        # print("1: ",num_token_masked)
        # print("2: ",steps)
        
        if sum(steps) != column_num:
            steps[-1] = column_num - sum(steps[:-1])
        # print("3: ",steps)
        assert sum(steps) == column_num, "Total steps must sum up to the number of columns"

        if self.fixed_test_mask:
            num_conditioned_columns = 5 if column_num == self.nc else 3
            
            if steps[0] < num_conditioned_columns:
                step0 = steps[0]
                steps[0] = 0
                steps[1] = steps[1] - (num_conditioned_columns-step0)
            elif steps[0] + steps[1] < num_conditioned_columns:
                raise AssertionError("The sum of the first two steps does not meet the minimum required number of conditioned columns.")
            else:
                steps[0] = steps[0] - num_conditioned_columns
        return steps

    def update_temperature(self, starting_temperature, steps_until_x0):
        return starting_temperature * (steps_until_x0 / self.n_iter)

    def generate_predictions(self, output):
        if self.sample_by in ["categorical", "greedy"]:
            filtered_logits = output["cat_pred"]
        elif self.sample_by == "gumbel":
            filtered_logits = top_k(output["cat_pred"], self.topk_filter_thres)

        cat_pred, _ = self.out2table.sample_categorical(filtered_logits, temperature=self.temperature, sample_by=self.sample_by)
        
        if self.boxcox_transformation:
            num_pred = self.out2table.sample_numerical(output, positive=False, based_ref=False, normalize_again=True)
        else:    
            num_pred = self.out2table.sample_numerical(output, positive=True, based_ref=True, normalize_again=True)
        
        if not self.null_sample:
            probability = torch.sigmoid(output['null_pred']/self.sigmoid_temperature)
            null_pred = (probability >= 0.5).long().squeeze(2)
        else:
            probability = torch.sigmoid(output['null_pred']/self.sigmoid_temperature)
            distribution = torch.distributions.Bernoulli(probability)
            null_pred = distribution.sample().long().squeeze(2)

        return cat_pred, num_pred, null_pred
    
    def finalize_output(self, kwargs):
        num_col = self.out2table.unnormalize(kwargs["num_input_ids"])
        cat_col = kwargs["cat_input_ids"].float()

        # Replace null ids with NaN
        cat_col[kwargs["cat_null_ids"]==1] = float('NaN')
        num_col[kwargs["num_null_ids"]==1] = float('NaN')

        return torch.cat([cat_col, num_col], dim=1)

    def update_predictions(self, kwargs, output, cat_pred, num_pred, null_pred, cat_step, num_step):
        for key_prefix, pred, step in zip(['cat', 'num'], [cat_pred, num_pred], [cat_step, num_step]):
            kwargs = self.update_prediction(kwargs, output, pred, null_pred, step, key_prefix)
        return kwargs
    
    def update_prediction(self, kwargs, output, pred, null_pred, step, key_prefix):
        mask = kwargs[key_prefix + '_input_labels'] != -100
        batch_size, col_num = pred.shape
        
        if key_prefix == 'cat':
            pred_prob = output['cat_pred']
            pred_prob += self.softmax_mask.to(mask.device)
            probs_without_temperature = torch.exp(self.softmax_layer(pred_prob))            
            scores = rearrange(probs_without_temperature.gather(2, pred[..., None]), '... 1 -> ...')
        else:
            scores = output['var_pred'].squeeze(dim=-1)
        
        if self.unmask_by == "random":
            sorted_indices = torch.stack([torch.randperm(col_num) for _ in range(batch_size)], dim=0).to(scores.device)
        elif self.unmask_by == "confidence":
            sorted_indices = torch.sort(scores, descending=(key_prefix == 'cat'))[1]
        elif self.unmask_by == "classimbalance":
            column_order = self.cat_column_order if key_prefix == 'cat' else self.num_column_order
            sorted_indices = torch.stack([torch.LongTensor(column_order) for _ in range(batch_size)], dim=0).to(scores.device)            
        elif self.unmask_by == "classimbalance&random":
            column_order = self.cat_column_order if key_prefix == 'cat' else self.num_column_order
            sorted_indices = torch.stack([torch.LongTensor(column_order) for _ in range(batch_size)], dim=0).to(scores.device)            
            sorted_indices = slight_shuffle(sorted_indices)
        else:
            raise AssertionError("Choose unmasking method...")

        assert sorted_indices.shape == mask.shape

        for batch_idx in range(batch_size):
            indices = torch.nonzero(mask[batch_idx]).squeeze()
            selected_indices = sorted_indices[batch_idx][torch.isin(sorted_indices[batch_idx], indices)][:step]

            new_mask = torch.zeros_like(mask[batch_idx])
            new_mask[selected_indices] = True
            # kwargs = self.update_inputs_and_labels(kwargs, key_prefix, batch_idx, new_mask, pred, null_pred)
            mask[batch_idx] = new_mask
        kwargs = self.update_inputs_and_labels(kwargs, key_prefix, mask, pred, null_pred)
        return kwargs

    def update_inputs_and_labels(self, kwargs, key_prefix, new_mask, pred, null_pred):       
        if key_prefix == 'cat':
            null_token_id = self.cat_null_token_id 
            null_pred_slice = null_pred[:, :self.nc]
            kwargs[key_prefix + "_input_ids"][new_mask] = pred[new_mask].long()
        else:
            null_token_id = self.num_null_token_id
            null_pred_slice = null_pred[:, self.nc:]
            kwargs[key_prefix + "_input_ids"][new_mask] = pred[new_mask]
        
        kwargs[key_prefix + "_input_labels"][new_mask] = -100        
        kwargs[key_prefix + "_input_ids"][new_mask & null_pred_slice.bool()] = null_token_id
        kwargs[key_prefix + '_null_ids'][new_mask] = null_pred_slice[new_mask]
        kwargs[key_prefix + '_null_labels'][new_mask] = null_pred_slice[new_mask]
        return kwargs
    
    def get_targets(self, sample):
        #unmaksed version
        targets = {}
        for input_type in ['cat', 'num']:
            targets[input_type+'_input_labels'] = sample['net_input'][input_type+'_input_labels']
            targets[input_type+'_null_labels'] = sample['net_input'][input_type+'_null_labels']
            targets[input_type+'_null_ids'] = sample['net_input'][input_type+'_null_ids']
        return targets

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

def slight_shuffle(input_matrix, shift=0.5, scale=0.3):    
    shuffled_matrix = torch.zeros_like(input_matrix)
    
    for idx, row in enumerate(input_matrix):
        # Normalize the tensor values to [0, 1] range
        max_val = len(row) - 1
        normalized_row = torch.arange(len(row)) / max_val
    
        # Create small perturbations in the range [-0.05, 0.05]
        random_perturbations = (torch.rand_like(normalized_row) - shift) * scale
        
        # Add perturbations to the normalized tensor
        perturbed_row = normalized_row + random_perturbations
        
        # Sort based on perturbed values to get the slightly shuffled order
        sorted_indices = perturbed_row.argsort()
        shuffled_matrix[idx] = row[sorted_indices]
    
    return shuffled_matrix