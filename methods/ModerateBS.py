from .SelectionMethod import SelectionMethod
import torch
import numpy as np
import os


# import copy

class ModerateBS(SelectionMethod):
    method_name = 'ModerateBS'

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


    def get_median(self, features, targets):
        # get the median feature vector of each class
        num_classes = len(np.unique(targets, axis=0))
        prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)

        for i in range(num_classes):
            prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
        return prot

    def get_features(self, inputs, targets, indexes):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model.eval()
        outputs, features = model.feat_nograd_forward(inputs)

        return features

    def get_distance(self, features, labels):
        labels = labels.cpu().numpy().tolist()
        features = features.cpu().numpy().tolist()
        features = np.array(features)
        labels = np.array(labels)
        prots = self.get_median(features, labels)
        prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))

        num_classes = len(np.unique(labels))
        for i in range(num_classes):
            prots_for_each_example[(labels == i).nonzero()[0], :] = prots[i]
        distance = np.linalg.norm(features - prots_for_each_example, axis=1)

        return distance

    def get_idx(self, radio, distance):
        rate = 1-radio
        low = 0.5 - rate / 2
        high = 0.5 + rate / 2

        sorted_idx = distance.argsort()
        low_idx = round(distance.shape[0] * low)
        high_idx = round(distance.shape[0] * high)

        ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))

        return ids


    def before_batch(self, i, inputs, targets, indexes, epoch):
        if self.iter_selection:
            ratio = self.get_ratio_per_epoch(epoch)
            if ratio == 1.0:
                if i == 0:
                    self.logger.info('using all samples')
                return super().before_batch(i, inputs, targets, indexes, epoch)
            else:
                if i == 0:
                    self.logger.info(f'balance: {self.balance}')
                    self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))
            selected_num_samples = int(inputs.shape[0] * ratio)
            features = self.get_features(inputs, targets, indexes)
            distance = self.get_distance(features, targets)
            indices = self.get_idx(ratio, distance)

            inputs = inputs[indices]
            targets = targets[indices]
            indexes = indexes[indices]
            return inputs, targets, indexes

        else:
            return super().before_batch(i, inputs, targets, indexes, epoch)