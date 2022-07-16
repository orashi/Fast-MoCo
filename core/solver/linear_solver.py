import argparse
import os
import pprint
import json
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from core.utils import dist as link
import random
import numpy as np

from tensorboardX import SummaryWriter
from easydict import EasyDict
from core.solver.ssl_solver import SSLSolver
from core.model import model_entry
from core.utils.dist import dist_init, broadcast_object
from core.utils.misc import (count_params, count_flops, load_state_model, AverageMeter, load_state_optimizer,
                             accuracy, makedir, create_logger, get_logger, modify_state)
from core.optimizer import optim_entry
from core.data import build_imagenet_train_dataloader, build_imagenet_test_dataloader


class LinearImageNetSolver(SSLSolver):
    def setup_env(self):
        # >>> dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size, self.dist.local_id = link.get_rank(), link.get_world_size(), link.get_local_rank()
        self.prototype_info.world_size = self.dist.world_size

        # >>> directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints_finetune')
        self.path.event_path = os.path.join(self.path.root_path, 'events_finetune')
        self.path.result_path = os.path.join(self.path.root_path, 'results_finetune')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)

        # >>> tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)

        # >>> logger
        create_logger(os.path.join(self.path.root_path, 'log_finetune.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")

        # >>> load pretrain checkpoint
        try:
            self.logger.info('======= Looking for local finetune pretrain... =======')
            local_ft_ckpt = os.path.join(self.path.root_path, 'checkpoints_finetune', 'ckpt.pth')
            self.state = torch.load(local_ft_ckpt, 'cpu')
            self.logger.info(f"Recovering from {local_ft_ckpt}, keys={list(self.state.keys())}")
        except:
            self.logger.info('======= Local finetune pretrain NOT FOUND =======')
            try:
                self.logger.info('======= Looking for local pretrain... =======')
                local_ft_ckpt = os.path.join(self.path.root_path, 'checkpoints', 'ckpt.pth')
                self.state = torch.load(local_ft_ckpt, 'cpu')
                self.logger.info(f"Recovering from {local_ft_ckpt}, keys={list(self.state.keys())}")
            except:
                self.logger.info('======= local pretrain NOT FOUND =======')
                self.logger.info('======= Looking for pretrain... =======')
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)

            state_dict = self.state['model']

            for k in list(state_dict.keys()):
                if 'backbone' in k and ('fc' not in k or self.config.model.type.startswith("vit")
                                        or self.config.model.type.startswith("swin")):  # rename & clean loaded keys
                    state_dict[k[len("module.backbone."):]] = state_dict[k]  # remove module.backbone.
                del state_dict[k]
            self.state = {'model': state_dict, 'last_iter': 0}

        # >>> seed initialization
        seed = self.config.get('seed', 233)
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
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # init the fc layer

        if self.config.model.type.startswith("vit") or self.config.model.type.startswith("swin"):
            self.model.head.weight.data.normal_(mean=0.0, std=0.01)
            self.model.head.bias.data.zero_()
        elif self.config.model.type.startswith("resnet"):
            self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.model.fc.bias.data.zero_()
        else:
            raise NotImplementedError

        self.model.cuda()
        self.prototype_info.model = self.config.model.type

        count_params(self.model)
        count_flops(self.model, input_shape=[1, 3, self.config.data.input_size, self.config.data.input_size])

        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                               device_ids=[self.dist.local_id],
                                                               output_device=self.dist.local_id,
                                                               find_unused_parameters=True)

        list_model_state_dict_keys = []
        for key in list(self.model.module.state_dict().keys()):
            if 'tracked' not in key:
                list_model_state_dict_keys.append(key)

        list_pretrain_state_dict_keys = []
        for key in list(self.state['model'].keys()):
            if 'tracked' not in key:
                list_pretrain_state_dict_keys.append(key)

        for k_state, k_model in zip(list_pretrain_state_dict_keys, list_model_state_dict_keys):
            if k_state != k_model:
                self.state['model'][k_model] = self.state['model'][k_state]
                del self.state['model'][k_state]
                self.logger.info(f"{k_state} ==> {k_model}, del {k_state}")

        load_state_model(self.model.module, self.state['model'])

    def build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        opt_config.kwargs.params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.config.get('bn_fc', False):
            assert len(opt_config.kwargs.params) == 4  # fc.weight, fc.bias, bn.weight, bn.bias
        else:
            assert len(opt_config.kwargs.params) == 2  # fc.weight, fc.bias

        self.optimizer = optim_entry(opt_config)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

    def build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.type == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
            self.val_data = build_imagenet_test_dataloader(self.config.data)
        else:
            raise NotImplementedError

    def pre_train(self):
        super(LinearImageNetSolver, self).pre_train()
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes

        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.pre_train()
        best_prec1 = 0
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        self.model.train()
        if self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False):
            self.model.module.fc[0].train()
        if self.config.get('bn_112_stat', False):
            self.model.module.bn1.change_current_res(112)

        for i, batch in enumerate(self.train_data['loader']):
            input = batch['image']
            target = batch['label']
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            target = target.squeeze().cuda().long()
            input = input.cuda()

            logits = self.model(input)

            loss = self.criterion(logits, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

            reduced_loss = loss.clone() / self.dist.world_size
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)

            self.optimizer.zero_grad()

            loss.backward()
            dist.barrier()
            self.optimizer.step()

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                curr_epoch = (curr_step - 1) // self.config.data.iter_per_epoch + 1
                log_msg = f'Epoch: [{curr_epoch}/{self.config.data.max_epoch}]\t' \
                          f'Iter: [{curr_step}/{total_step}]|[{curr_step - (curr_epoch - 1) * self.config.data.iter_per_epoch}/{self.config.data.iter_per_epoch}]\t' \
                          f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                          f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                          f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                          f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                          f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                          f'LR {current_lr:.4f}\t' \
                          f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            if (curr_step % self.config.saver.val_freq == 0 or curr_step == total_step) and curr_step > 0:
                metrics = self.evaluate()
                best_prec1 = max(metrics.metric['top1'], best_prec1)
                # testing logger
                if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                    metric_key = 'top{}'.format(self.topk)
                    self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                    self.tb_logger.add_scalar('acc5_val', metrics.metric[metric_key], curr_step)
                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    torch.save(self.state, ckpt_name)
                    self.sanity_check(self.model.module.state_dict(), self.state['model'])

            end = time.time()
        self.logger.info('best acc:' + str(best_prec1) + '\n')

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            input = batch['image']
            label = batch['label']
            input = input.cuda()
            label = label.squeeze().view(-1).cuda().long()
            # compute output
            logits = self.model(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, batch)

        writer.close()
        dist.barrier()
        if self.dist.rank == 0:
            metrics = self.val_data['loader'].dataset.evaluate(res_file)
            self.logger.info(json.dumps(metrics.metric, indent=2))
        else:
            metrics = {}
        dist.barrier()
        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        if self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False):
            self.model.module.fc[0].train()
        self.model.train()
        return metrics

    def test(self):
        self.pre_train()
        metrics = self.evaluate()
        self.logger.info('acc:' + str(metrics.metric['top1']) + '\n')

    def sanity_check(self, state_dict, pretrained_weights):
        """
        Linear classifier should not change any weights other than the linear layer.
        This sanity check asserts nothing wrong happens (e.g., BN stats updated).
        """
        # print("=> loading '{}' for sanity check".format(pretrained_weights))
        # checkpoint = torch.load(pretrained_weights, map_location="cpu")
        # state_dict_pre = checkpoint['model']

        list_model_state_dict_keys = []
        for key in list(state_dict.keys()):
            if 'tracked' not in key:
                list_model_state_dict_keys.append(key)

        list_pretrain_state_dict_keys = []
        for key in list(pretrained_weights.keys()):
            if 'tracked' not in key:
                list_pretrain_state_dict_keys.append(key)

        for k, k_pre in zip(list_model_state_dict_keys, list_pretrain_state_dict_keys):
            # only ignore fc layer
            if 'fc.weight' in k or 'fc.bias' in k:
                continue
            if 'num_batches_tracked' not in k:
                continue
            # if (self.config.get('bn_fc', False) or self.config.get('bnnaf_fc', False)) and 'fc.' in k:
            if self.config.get('bn_fc', False) and 'fc.' in k:
                continue
            # name in pretrained model
            # k_pre = 'module.features.' + k[len('module.'):] \
            #     if k.startswith('module.') else 'module.features.' + k
            assert ((state_dict[k].cpu() == pretrained_weights[k_pre]).all()), \
                '{} is changed in linear classifier training.'.format(k)

        print("=> sanity check passed.")


def main():
    parser = argparse.ArgumentParser(description='linear solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument("--tcp_port", type=str, default="5671")

    args = parser.parse_args()

    dist_init(port=str(args.tcp_port))
    # build solver
    solver = LinearImageNetSolver(args.config)

    if solver.config.data.last_iter < solver.config.data.max_iter:
        solver.train()
    else:
        solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
