import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

"""
Statistial test
- positive sampling (X)
- unmasked -> NaN
- null -> NaN
Generated data (used for TRTS prediction)
- positive sampling (O)
- unmaksed -> original value
- null -> NaN
Also, we have reference values for specific columns
"""

class MMTGOut2Table(nn.Module):
    def __init__(self, _config):
        super().__init__()
        self.config = _config
        self.gen_table = []
        # self.gen_pos_table = []
        self.data_root = _config['input_path']
        self.znorm_df = pd.read_csv(os.path.join(self.data_root, 'znorm.csv'))
        self.column_order = self.znorm_df.iloc[:, 0].values        
        assert isinstance(self.column_order[0], str)

        # create softmax mask
        cat_nunique = _config['cat_columns_numclass_list']
        self.softmax_mask = torch.full((_config['nc'], _config['cat_vocab_size']), -100000)#.to('cuda')
        cumulated = 0
        for s, v in enumerate(cat_nunique):
            self.softmax_mask[s, cumulated:cumulated+v] = 0
            cumulated += v
        self.softmax_layer = nn.LogSoftmax(dim=-1)
        self.test_sets = _config["test_sets"]
        self.sample_by= _config["sample_by"]
        self.mask_scheduler = _config["mask_scheduler"]

    def forward(self, x, **kwargs):
        cat_pred_tb, _ = self.sample_categorical(x, sample_by=self.sample_by)
        cat_pred_pos_tb = cat_pred_tb.clone()
        num_pred_tb = self.sample_numerical(x)
        num_pred_pos_tb = self.sample_numerical(x, positive=True, based_ref=True) 

        # Replace predicted null with NaN
        null_pred = torch.argmax(x['null_pred'], dim=-1)
        null_cat_pred, null_num_pred = null_pred[:, :cat_pred_tb.shape[1]], null_pred[:, cat_pred_tb.shape[1]:]

        cat_pred_tb[null_cat_pred==1] =  float('NaN')
        cat_pred_pos_tb[null_cat_pred==1] =  float('NaN')
        num_pred_tb[null_num_pred==1] =  float('NaN')
        num_pred_pos_tb[null_num_pred==1] =  float('NaN')

        # Replace unmasked with NaN
        num_unmaksed = kwargs['num_input_labels'] == -100
        cat_unmaksed = kwargs['cat_input_labels'] == -100
        num_pred_tb[num_unmaksed] = float('NaN')
        cat_pred_tb[cat_unmaksed] = float('NaN')
        
        # Create table concatenating categorical and numerical columns
        self.gen_table.append(torch.cat([cat_pred_tb, num_pred_tb], dim=1))
        # self.gen_pos_table.append(torch.cat([cat_pred_pos_tb, num_pred_pos_tb], dim=1))

    # def create_label_table(self):
    #     input_ids = np.load(os.path.join(self.config["input_path"], "unnormalized_input_ids.npy"))
    #     fold = pd.read_csv(os.path.join(self.config["input_path"], "fold", f"snuh_{self.config['seed']}_fold_split.csv"))

    #     if self.test_sets[0] =="test":
    #         test_split = fold.fold.values == 0
    #     elif len(self.test_sets) == 2:
    #         test_split = fold.fold.values != 0
    #     elif self.test_sets[0] =="valid":
    #         test_split = fold.fold.values == 2
    #     else:
    #         AssertionError("Check")

    #     num_table = self.config["nc"] + self.config["nn"]
    #     cat_null_ids = self.config["cat_null_id"]
    #     num_null_ids = self.config["num_null_id"]
    #     cat_col_num = self.config["nc"]

    #     df = pd.DataFrame(data=input_ids[:, :num_table], columns=self.config["column_order"])
    #     df = df.iloc[:, :num_table]
    #     df = df.iloc[test_split]

    #     df.iloc[:, :cat_col_num] = df.iloc[:, :cat_col_num].replace(cat_null_ids, float("NaN"))
    #     df.iloc[:, cat_col_num:] = df.iloc[:, cat_col_num:].replace(num_null_ids, float("NaN"))
    #     return df
    

    def sample_categorical(self, x, temperature=1.0, sample_by="categorical"):
        if sample_by == "categorical":
            if isinstance(self.mask_scheduler, str):
                cat_pred_prob = x
                cat_pred_prob += self.softmax_mask.to(x.device)
                cat_pred_prob = torch.exp(self.softmax_layer(cat_pred_prob))
                cat_pred = torch.distributions.Categorical(cat_pred_prob).sample().long()
            else:
                cat_pred_prob = x['cat_pred']
                cat_pred_prob += self.softmax_mask.to(x['cat_pred'].device)

                cat_pred_prob = torch.exp(self.softmax_layer(cat_pred_prob))
                cat_pred = torch.distributions.Categorical(cat_pred_prob).sample().float()

        elif sample_by == "gumbel":
            if isinstance(self.mask_scheduler, str):
                cat_pred = gumbel_sample(x, temperature)
            else:
                cat_pred = gumbel_sample(x['cat_pred'], temperature)

        elif sample_by == "greedy":
            cat_pred = torch.argmax(x['cat_pred'], dim=-1).float()

        return cat_pred, None
    

    def sample_numerical(self, x, num_null_labels=None, positive=False, based_ref=False, normalize_again=False):
        num_pred = torch.normal(x['mean_pred'], x['var_pred'].sqrt()).squeeze(dim=-1)
        num_pred = self.unnormalize(num_pred, num_null_labels=num_null_labels)

        if positive:
            num_pred = self.positive_sampling(x, num_pred)
        if based_ref:
            num_pred = self.reference_sampling(x, num_pred)
        if normalize_again:
            num_pred = self.normalize(num_pred, num_null_labels=num_null_labels)    
        return num_pred
    

    def normalize(self, num_col, num_null_labels=None):
        mean = torch.Tensor(self.znorm_df['mean'].values).cuda()
        std = torch.Tensor(self.znorm_df['std'].values).cuda()

        if num_null_labels is not None:
            num_null_loc = num_null_labels == 1
            num_col = (num_col - mean) / std * (~num_null_loc)
        else:
            num_col = (num_col - mean) / std
        return num_col


    def unnormalize(self, num_col, num_null_labels=None):        
        mean = torch.Tensor(self.znorm_df['mean'].values).cuda()
        std = torch.Tensor(self.znorm_df['std'].values).cuda()
        
        if num_null_labels is not None:
            num_null_loc = num_null_labels== 1
            num_col = (num_col * std + mean)*(~num_null_loc)
        else:
            num_col = num_col * std + mean
        return num_col


    def positive_sampling(self, x, num_pred):
        count = 0
        neg_mask = num_pred < 0
        while neg_mask.any():
            count += 1
            num_pred[neg_mask] = self.unnormalize(torch.normal(x['mean_pred'], x['var_pred'].sqrt()).squeeze(dim=-1))[neg_mask]
            neg_mask = num_pred < 0
            if count == 50:
                # print(f"Even after {count} iterations, {neg_mask.sum()} negative values occur.")
                break
        return num_pred
    

    def reference_sampling(self, x, num_pred):
        reference_values = {
            'A_SBP':[50,260],
            'A_DBP':[30,240],
            'A_HR':[30,140],
            'A_HT':[70,250], 
            'A_WT':[20,150], 
            'A_Age':[18,100],
        }

        for col_name, (min, max) in reference_values.items():
            col_index= list(self.column_order).index(col_name)
            
            count = 0 
            ood_mask = (num_pred[:, col_index] < min) | (num_pred[:, col_index] > max)
            while ood_mask.any():
                count += 1
                num_pred[ood_mask] = self.unnormalize(torch.normal(x['mean_pred'], x['var_pred'].sqrt()).squeeze(dim=-1))[ood_mask]
                ood_mask = (num_pred[:, col_index] < min) | (num_pred[:, col_index] > max)
                if count == 50:
                    # print(f"Even after {count} iterations, {ood_mask.sum()} outliers occur.")
                    break
        return num_pred


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)
