from .SelectionMethod import SelectionMethod
import torch
import numpy as np
import os


# import copy

class CCSBS(SelectionMethod):
    method_name = 'CCSBS'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.iter_selection = config['method_opt']['iter_selection'] if 'iter_selection' in config[
            'method_opt'] else False
        self.epoch_selection = config['method_opt']['epoch_selection'] if 'epoch_selection' in config[
            'method_opt'] else False
        assert (self.iter_selection and not self.epoch_selection) or (
                    not self.iter_selection and self.epoch_selection), 'there should be one and only one True in iter_selection and epoch_selection'

        self.num_epochs_per_selection = config['method_opt'][
            'num_epochs_per_selection'] if 'num_epochs_per_selection' in config['method_opt'] else 1

        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config[
            'method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0

        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False

    def get_ratio_per_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            self.logger.info('warming up')
            return 1.0
        if self.ratio_scheduler == 'constant':
            return self.ratio
        elif self.ratio_scheduler == 'increase_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'decrease_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'increase_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        elif self.ratio_scheduler == 'decrease_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError

    def before_batch(self, i, inputs, targets, indexes, epoch):
        # to do
