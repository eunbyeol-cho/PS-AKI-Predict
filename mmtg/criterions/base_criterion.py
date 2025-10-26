import torch
import torch.nn as nn


class BaseCriterion():
    def __init__(self, _config):
        self.cat_null_token_id = _config['cat_null_id']
        self.num_null_token_id = _config['num_null_id']
        self.mask_token_id = self.cat_null_token_id + 1

        self.cat_criterion = nn.NLLLoss(ignore_index=-100)
        self.num_criterion = nn.GaussianNLLLoss()
        # self.null_criterion = nn.CrossEntropyLoss()
        self.null_criterion = nn.BCEWithLogitsLoss()
        self.num_loss_weight = _config['num_loss_weight']

        self.mse_error = nn.MSELoss()


    @property
    def compare(self):
        raise NotImplementedError


    def reset(self):
        raise NotImplementedError


    def __call__(self):
        raise NotImplementedError