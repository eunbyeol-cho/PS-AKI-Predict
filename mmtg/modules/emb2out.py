import torch
import torch.nn as nn


class MMTGEmb2Out(nn.Module):
    def __init__(self, _config):
        super().__init__()

        self.test_only = _config['test_only']

        self.use_text = 'text' in _config['modality']

        bert_config = {
            "tiny": (2, 128, 2),
            "mini": (4, 256, 4),
            "small": (4, 512, 8),
            "medium": (8, 512, 8),
            "base": (12, 768, 12),
        }

        _, self.emb_dim, _ = bert_config[_config['bert_type']]
    
        self.nc = _config['nc']
        self.nn = _config['nn']
        self.ntable = self.nc + self.nn
        self.ntext = _config['ntext']

        #Variable settings
        self.sep_regressor = _config['sep_regressor']
        self.var_head_type = _config['var_head_type']
        self.relu = nn.ReLU()  
        self.eps = 1e-6

        #Construct head
        # self.null_head = nn.Linear(self.emb_dim, 2)
        self.null_head = nn.Linear(self.emb_dim, 1)
        self.mean_head = nn.Parameter(torch.randn(_config['nn'], self.emb_dim)) if self.sep_regressor else nn.Linear(self.emb_dim, 1)
        self.var_head = nn.Parameter(torch.randn(_config['nn'], self.emb_dim)) if self.sep_regressor else nn.Linear(self.emb_dim, 1)
        self.cat_head = nn.Linear(self.emb_dim, _config['cat_vocab_size'])

        cat_nunique = _config['cat_columns_numclass_list']
        # self.softmax_mask = torch.full((_config['nc'], _config['cat_vocab_size']), -100000).to('cuda')
        # cumulated = 0
        # for s, v in enumerate(cat_nunique):
        #     self.softmax_mask[s, cumulated:cumulated+v] = 0
        #     cumulated += v
        self.softmax_layer = nn.LogSoftmax(dim=-1)


    def forward(self, x, **kwargs):
        #Null head
        null_pred = self.null_head(x[:, :self.ntable])

        #Categorical head
        cat_pred = self.cat_head(x[:, :self.nc])
        if self.test_only:
            cat_pred = cat_pred
            # cat_pred += self.softmax_mask
        else:
            cat_pred = self.softmax_layer(cat_pred)
        
        #Numerical head
        if self.sep_regressor:
            mean_pred = torch.sum(torch.mul(x[:, self.nc:self.ntable], self.mean_head), axis=-1).unsqueeze(-1) 
            var_pred = torch.sum(torch.mul(x[:, self.nc:self.ntable], self.var_head), axis=-1).unsqueeze(-1)
        else:
            mean_pred = self.mean_head(x[:, self.nc:self.ntable])
            var_pred = self.var_head(x[:, self.nc:self.ntable])

        if self.var_head_type == 'abs':
            var_pred = abs(var_pred)
        elif self.var_head_type == 'relu':
            var_pred = self.relu(var_pred) + self.eps
        else:
            raise ValueError

        return {
            'cat_pred':cat_pred,
            'mean_pred': mean_pred,
            'var_pred': var_pred,
            'null_pred': null_pred
        }