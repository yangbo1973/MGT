import os
from abc import ABCMeta, abstractmethod

import torch


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None, test_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []
        self.test_loggers = test_loggers if test_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)

    def log_test(self, log_data):
        for logger in self.test_loggers:
            logger.log(**log_data)


class AbstractBaseLogger(metaclass=ABCMeta): # 带抽象方法的就是抽象类
    @abstractmethod # 继承抽象类的类必须要重写抽象方法
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, metric_key='mean_iou', filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model".format(self.metric_key))
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename)


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['epoch'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['epoch'])

    def complete(self, *args, **kwargs):
        self.writer.close()
