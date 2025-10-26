import os
import tqdm
import logging
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from mmtg.criterions import MLMCriterion
import mmtg.utils.trainer_utils as utils
from mmtg.datamodules.postprocess import minmax_clip
import mmtg.utils.eval_pipeline_six_hosp as ep

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, model, data_loaders, _config):

        self.config = _config
        
        # Load config from JSON (new format) or fall back to pickle (old format)
        import json
        config_json_path = os.path.join(self.config["input_path"], "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                self.info = json.load(f)
        else:
            # Fall back to pickle for backward compatibility
            self.info = pd.read_pickle(os.path.join(self.config["input_path"], "info_dict.pickle"))
        self.model = nn.DataParallel(model).to('cuda')
        self.data_loaders = data_loaders
        utils.count_parameters(self.model)

        #Training settings
        self.criterion = MLMCriterion(_config)
        self.n_epochs = self.config['n_epochs']
        self.lr =  self.config['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.patience = self.config['patience']

        exp_component = [
            ''.join(self.config['modality']),
            self.config['modeling'],
            self.config['mask_scheduler'],
            self.config['num_loss_weight'],
            self.config['bert_type'],
            self.config['var_head_type'],
            self.config['sep_regressor'],
            self.config['sep_numemb'],
            self.config['lr'],
            self.config['dropout'],
            self.config['seed'],
            ]
        
        ckpt_name = '_'.join(list(map(str, exp_component)))
        self.path = os.path.join(self.config['output_path'], ckpt_name)
        os.makedirs(self.config['output_path'], exist_ok=True)

        #Wandb
        if not self.config['debug']:
            wandb.init(
                project=self.config['wandb_project_name'],
                entity="emrsyn",
                config=self.config,
                reinit=True
            )
            wandb.run.name = ckpt_name

        if _config["target_study"]:
            fold_path = os.path.join(_config['input_path'], f'fold/snuh_{self.config["seed"]}_fold_split.csv')
            splits = pd.read_csv(fold_path)['fold'].values
            idcs = np.where(splits != 0)[0]
            self.src_samples = len(idcs)

            fold_path = os.path.join(_config['input_path'].replace(_config["study"], _config["target_study"]),
                                    f'fold/snuh_{_config["seed"]}_fold_split.csv')
            splits = pd.read_csv(fold_path)['fold'].values
            idcs = np.where(splits != 0)[0]
            self.trg_samples = len(idcs)

    def train(self):
        self.early_stopping = utils.EarlyStopping(
            patience=self.patience, 
            compare=self.criterion.compare,
            metric=self.criterion.update_target
            )

        for epoch in range(self.n_epochs):
            self.model.train()
            for sample in tqdm.tqdm(self.data_loaders['train']):
                self.optimizer.zero_grad(set_to_none=True)

                targets = self.model.module.get_targets(sample)
                net_output  = self.model(**sample['net_input'])
                
                loss = self.criterion('loss', net_output, targets)
                
                # if torch.isnan(loss['total_loss']).item():
                loss['total_loss'].backward()
                self.optimizer.step()
                                
                with torch.no_grad():
                    acc = self.criterion('acc', net_output, targets)

            with torch.no_grad():
                epoch_log = self.criterion.get_epoch_dict(len(self.data_loaders['train']))

            summary = utils.log_from_dict(epoch_log, 'train', epoch)
            if not self.config['debug']:
                wandb.log(summary)
            
            should_stop = self.validate(epoch)
            if should_stop:
                break

        self.test()
        if not self.config['debug']:
            wandb.finish(0)


    def inference(self, epoch, subsets):
        self.model.eval()

        self.table_syn = []
        self.table_real = ep.create_label_table(self.config["input_path"], self.info, self.config["seed"], dataset_type="trainvalid")

        with torch.no_grad():
            for subset in subsets:
                print(f"Inference on {subset}")
                for sample in tqdm.tqdm(self.data_loaders[subset]):
                    targets = self.model.module.get_targets(sample)
                    
                    # if self.config['iter_decoding'] and self.config['test_only'] and self.config["target_study"]:
                    #     net_output = self.process_inference_batches(sample)
                    # elif self.config['iter_decoding'] and self.config['test_only'] and (self.config["target_study"] is None):
                    #     net_output, _ = self.model.module.iterative_decode(**sample['net_input'])
                    if self.config['iter_decoding'] and self.config['test_only']:
                        net_output, _ = self.model.module.iterative_decode(**sample['net_input'])
                    else:
                        net_output = self.model(**sample['net_input'])

                    loss = self.criterion('loss', net_output, targets)
                    acc = self.criterion('acc', net_output, targets)

                epoch_log = self.criterion.get_epoch_dict(len(self.data_loaders[subset]))
                summary = utils.log_from_dict(epoch_log, subset, epoch)
                if not self.config['debug']:
                    wandb.log(summary)

        return epoch_log

    def validate(self, epoch):
        break_token = False
        epoch_log = self.inference(epoch, ['valid'])
        
        if self.early_stopping(epoch_log[self.criterion.update_target]):
            utils.model_save(self.path, self.model, self.optimizer, epoch)

        if self.early_stopping.early_stop:
            logger.info(f'Early stopped! All valid finished at {epoch} ...')
            break_token=True
        return break_token

    def process_generated_table(self, n_patience):
        # Convert tensor output to DataFrame
        table_gen = pd.DataFrame(torch.cat(self.model.module.gen_table).cpu().numpy(),
                                 columns=self.config['column_order'])
        # Apply min-max clipping
        _, mask = minmax_clip(df=table_gen, real_df=self.table_real, config=self.config)
        table_gen = table_gen[~mask]

        # Compare value counts to check data quality
        if self.config["fixed_test_mask"] and self.config["spark_features"]:
            return table_gen
        else:
            value_counts_real = self.table_real['O_AKI'].value_counts()
            value_counts_gen = table_gen['O_AKI'].value_counts()
            if value_counts_gen.min() * n_patience < value_counts_real.min():
                raise AssertionError("Low possibility to make balanced dataframe")
            return table_gen

    def check_table_is_sufficient(self, iteration, table_gen):
        value_counts_real = self.table_real['O_AKI'].value_counts(normalize=False)
        try:
            aki_table_gen = self.balance_sample_with_exact_counts(table_gen, value_counts_real)
            return aki_table_gen, True 
        except ValueError as e:
            print(f"Iteration {iteration}: Unable to balance samples adequately: {e}")
            return None, False

    def balance_sample_with_exact_counts(self, df, target_counts):
        result_df = pd.DataFrame(columns=df.columns)
        for label, count in target_counts.items():
            subset = df[df['O_AKI'] == label]
            if subset.shape[0] < count:
                raise ValueError(f"Not enough samples for label {label}: needed {count}, but found {subset.shape[0]}")
            sampled_subset = subset.sample(n=count, random_state=0).reset_index(drop=True)
            result_df = pd.concat([result_df, sampled_subset])
        return result_df

    def test(self):
        epoch, self.model, _ = utils.model_load(self.path, self.model, self.optimizer)
        table_gen = None
        aki_table_gen = None

        if self.config['iter_decoding'] and self.config['test_only']:
            n_patience = 10
            for i in range(n_patience):
                epoch_log = self.inference(epoch, self.config['test_sets'])
                table_gen = self.process_generated_table(n_patience)
                
                if self.config['target_study']:
                    check_sufficient_table = len(table_gen) > 64000
                else:
                    _, check_sufficient_table = self.check_table_is_sufficient(i, table_gen)
                if check_sufficient_table:
                    break
        else:
            epoch_log = self.inference(epoch, self.config['test_sets'])
        
        if self.config['test_only']:
            if self.config["target_study"] is None:
                table_real = self.table_real
                table_gen = self.process_generated_table(n_patience)
                aki_table_gen, check_sufficient_table = self.check_table_is_sufficient(i, table_gen)
                assert check_sufficient_table, "Check for sufficient table failed!"
            else:
                table_real = self.table_real
                # aki_table_gen = pd.concat(self.table_syn)
                aki_table_gen =  self.process_generated_table(n_patience)
            
            # Debug output for table dimensions
            print(f"table_real shape: {table_real.shape}")
            print(f"aki_table_gen shape: {aki_table_gen.shape}")

            dataset_used = "".join(self.config["test_sets"])
            self.path += f"_{self.config['n_iter']}"
            self.path += f"_{self.config['sample_by']}"
            self.path += f"_{self.config['unmask_by']}"
            self.path += f"_{self.config['null_sample']}"
            self.path += f"_{self.config['sigmoid_temperature']}"
            self.path += f"_{dataset_used}"

            if self.config["target_study"]:
                self.path += f"_{self.config['target_study']}"
            
            output_file = f"{self.path}_gen_em.csv" if self.config['fixed_test_mask'] else f"{self.path}_gen_constaki.csv"
            aki_table_gen.to_csv(output_file, index=False)
            print(f'Generated dataframes are saved at: {output_file}')
            # ep.eval_pipeline(self.config, f'{self.path}_gen.csv')

        return epoch_log

        
