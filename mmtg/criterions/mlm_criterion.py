import torch
import torch.nn as nn
from .base_criterion import BaseCriterion


class MLMCriterion(BaseCriterion):
    def __init__(self, _config):
        super().__init__(_config)
        self.test_only = _config['test_only']
        self.reset()


    def reset(self):
        self.loss = {
            'total':0,
            'cat':0,
            'num':0,
            'null':0
        }
        self.acc = {
            'cat':0,
            'null':0
        }


    def __call__(self, criterion,  x, targets):
        iter_log = self.compute_mlm(criterion, x, targets)

        if criterion == 'loss':
            for loss_type in self.loss.keys():
                self.loss[loss_type] += iter_log[f'{loss_type}_loss']
        elif criterion == 'acc':
            for acc_type in self.acc.keys():
                self.acc[acc_type] += iter_log[f'{acc_type}_acc']
                
        return iter_log

    def get_epoch_dict(self, total_iter):
        epoch_log = {}
        for loss_type in self.loss.keys():
            epoch_log[f'{loss_type}_loss'] = self.loss[loss_type] / total_iter
        for acc_type in self.acc.keys():
            epoch_log[f'{acc_type}_acc'] = self.acc[acc_type] / total_iter
        self.reset()
        return epoch_log


    def compute_mlm(self, criterion,  x, targets):

        device = x['cat_pred'].device
        num_label = targets['num_input_labels'].to(device)
        mean_pred = x['mean_pred']
        var_pred =  x['var_pred']

        cat_label = targets['cat_input_labels'].to(device)
        cat_pred = x['cat_pred']

        null_label = torch.cat([targets['cat_null_labels'], targets['num_null_labels']], axis=1).to(device)
        null_pred = x['null_pred']

        #Should be excluded when calculating losses
        #ignored: if unmasked or null <-> masked and not null
        num_ignored = (num_label == -100) | (targets['num_null_labels'].to(device)==1)
        cat_ignored = (cat_label == -100) | (targets['cat_null_labels'].to(device)==1)
        masked = torch.cat([targets['cat_null_ids']==2, targets['num_null_ids']==2], axis=1).to(device) #unknown

        if criterion == 'loss':
           
            #Calculate loss
            null_loss = self.null_criterion(null_pred[masked].squeeze(1), null_label[masked].float())    
            cat_loss = self.cat_criterion(cat_pred[~cat_ignored], cat_label[~cat_ignored])  
            num_loss = self.num_criterion(mean_pred[~num_ignored], num_label[~num_ignored].unsqueeze(-1), var_pred[~num_ignored])
            
            loss = cat_loss + self.num_loss_weight * num_loss + null_loss

            if torch.isnan(loss).item():
                cat_loss = torch.tensor(0.0, device=device)
                num_loss = torch.tensor(0.0, device=device)
                null_loss = torch.tensor(0.0, device=device)
                loss = cat_loss + self.num_loss_weight * num_loss + null_loss

            return {
                'total_loss':loss,
                'null_loss':null_loss,
                'cat_loss':cat_loss,
                'num_loss':num_loss,
            }

        elif criterion == 'acc':
            #Calculate acc
            # null_pred_argmax = torch.argmax(null_pred, dim=-1)
            null_pred_sigmoid = torch.sigmoid(null_pred.squeeze(2))
            null_pred_argmax = (null_pred_sigmoid >= 0.5).long()

            correct = (null_label[masked]==null_pred_argmax[masked]).sum().detach().cpu()
            if len(null_label[masked]) == 0:
                null_acc = 1
            else:
                null_acc = int(correct)/len(null_label[masked])
            
            cat_pred_argmax = torch.argmax(cat_pred, dim=-1)
            correct = (cat_label[~cat_ignored]==cat_pred_argmax[~cat_ignored]).sum().detach().cpu()
            if len(cat_label[~cat_ignored]) == 0:
                cat_acc = 1
            else:
                cat_acc = int(correct)/(len(cat_label[~cat_ignored])) 
            return {
                'null_acc':null_acc,
                'cat_acc':cat_acc
            }


    @property
    def compare(self):
        return 'decrease' if 'loss' in self.update_target else 'increase'

    @property
    def update_target(self):
        return 'total_loss'