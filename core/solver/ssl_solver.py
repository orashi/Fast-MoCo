import os
import pprint

import torch
from core.utils import dist as link
import random
import numpy as np

from tensorboardX import SummaryWriter
from easydict import EasyDict

from core.utils.misc import AverageMeter, parse_config, makedir, create_logger, get_logger, modify_state
from core.lr_scheduler import scheduler_entry


class SSLSolver(object):

    def __init__(self, config_file):
        self.config_file = config_file
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_data()
        self.build_lr_scheduler()

    def setup_env(self):
        # >>> dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size, self.dist.local_id = link.get_rank(), link.get_world_size(), link.get_local_rank()
        self.prototype_info.world_size = self.dist.world_size

        # >>> directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)

        # >>> tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)

        # >>> logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")

        # >>> load pretrain checkpoint
        try:
            self.logger.info('======= Looking for local pretrain... =======')
            local_ft_ckpt = os.path.join(self.path.root_path, 'checkpoints', 'ckpt.pth')
            self.state = torch.load(local_ft_ckpt, 'cpu')
            self.logger.info(f"Recovering from {local_ft_ckpt}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
        except:
            self.logger.info('======= local pretrain NOT FOUND =======')
            try:
                self.logger.info('======= Looking for pretrain... =======')
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
                if hasattr(self.config.saver.pretrain, 'ignore'):
                    self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
            except:
                self.logger.info('======= pretrain NOT FOUND =======')
                self.state = {}
                self.state['last_iter'] = 0

        # >>> seed initialization
        seed = self.config.get('seed', 233)
        if self.config.get('dist_seed', False):
            seed += self.dist.rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # >>> reproducibility config    note: deterministic would slow down training
        if self.config.get('strict_reproduceable', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    def build_model(self):
        raise NotImplementedError

    def build_optimizer(self):
        raise NotImplementedError

    def build_data(self):
        raise NotImplementedError

    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)

        self.model.train()
        self.criterion = None

    def train(self):
        raise NotImplementedError
