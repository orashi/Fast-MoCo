import copy
import argparse
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist

from core.solver.ssl_solver import SSLSolver
from core.model import model_entry
from core.utils import unsupervised_entry
from core.utils.dist import dist_init
from core.utils.misc import count_params, count_flops, load_state_model, AverageMeter, load_state_optimizer
from core.optimizer import optim_entry
from core.data import build_imagenet_train_dataloader
from core.loss_functions import loss_entry


class ImageNetSolver(SSLSolver):
    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        if not self.config.model.kwargs.get('bn', False):
            self.config.model.kwargs.bn = copy.deepcopy(
                self.config.backbone.kwargs.bn)  # note: prevent anticipated inplace edit in model_entry.
            if self.config.model.kwargs.get('ema', False):
                bb_config_back = copy.deepcopy(self.config.backbone)
        else:
            if self.config.model.kwargs.get('ema', False):
                bb_config_back = copy.deepcopy(self.config.backbone)

        self.config.model.kwargs.backbone = model_entry(self.config.backbone)
        if self.config.model.kwargs.get('ema', False):
            if bb_config_back.kwargs.get('img_size', False):
                bb_config_back.kwargs.img_size = 224
            if bb_config_back.type in ['swin_t'] or bb_config_back.type.startswith("vit"):
                bb_config_back.kwargs.is_teacher = True

            self.config.model.kwargs.backbone_ema = model_entry(bb_config_back)

        self.model = unsupervised_entry(self.config.model)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()
        self.prototype_info.model = f"{self.config.model.type} + {self.config.backbone.type}"

        count_params(self.model.encoder)
        count_flops(self.model.encoder, input_shape=[1, 3, self.config.data.input_size, self.config.data.input_size])

        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                               device_ids=[self.dist.local_id],
                                                               output_device=self.dist.local_id,
                                                               find_unused_parameters=True)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        param_group = [
            {
                "params": [p for n, p in self.model.named_parameters() if not ("predictor" in n) and p.requires_grad],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if ("predictor" in n) and p.requires_grad],
            }
        ]

        opt_config.kwargs.params = param_group

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
        else:
            raise NotImplementedError

    def pre_train(self):
        super(ImageNetSolver, self).pre_train()
        self.meters.distance_z = AverageMeter(self.config.saver.print_freq)
        self.criterion = loss_entry(self.config.criterion)

    def train(self):
        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()

        for i, batch in enumerate(self.train_data['loader']):
            input = batch['image']  # [bs, #channel * 2, h, w]
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr, head_lr = self.lr_scheduler.get_lr()[0], self.lr_scheduler.get_lr()[1]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            input = input.cuda()

            # input -> p1, z1, p2, z2
            output = self.model(input)

            loss = self.criterion.forward(*output)
            loss = loss

            with torch.no_grad():
                distance_z = self.criterion.cosine_similarity(output[1], output[3])
            reduced_loss, reduced_dist_z = loss.clone() / self.dist.world_size, distance_z.clone() / self.dist.world_size
            self.meters.losses.reduce_update(reduced_loss)
            self.meters.distance_z.reduce_update(reduced_dist_z)

            self.optimizer.zero_grad()
            loss.backward()
            dist.barrier()

            if self.config.get('clip_grad_norm', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            self.optimizer.step()

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('distance_z_train', self.meters.distance_z.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                self.tb_logger.add_scalar('lr_head', head_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                curr_epoch = (curr_step - 1) // self.config.data.iter_per_epoch + 1

                log_msg = f'Epoch: [{curr_epoch}/{self.config.data.max_epoch}]\t' \
                          f'Iter: [{curr_step}/{total_step}]|[{curr_step - (curr_epoch - 1) * self.config.data.iter_per_epoch}/{self.config.data.iter_per_epoch}]\t' \
                          f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                          f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                          f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                          f'Distance_Z {self.meters.distance_z.val:.4f} ({self.meters.distance_z.avg:.4f})\t' \
                          f'LR {current_lr:.4f}\t' \
                          f'Head LR {head_lr:.4f}\t' \
                          f'Remaining Time {remain_time} ({finish_time})'

                self.logger.info(log_msg)

            if self.dist.rank == 0 and (
                    curr_step % self.config.saver.val_freq == 0 or curr_step == total_step) and curr_step > 0:
                ckpt_name = f'{self.path.save_path}/ckpt.pth'
                self.state['model'] = self.model.state_dict()
                self.state['optimizer'] = self.optimizer.state_dict()
                self.state['last_iter'] = curr_step
                torch.save(self.state, ckpt_name)
                if self.config.saver.save_many is True:
                    ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth'
                    torch.save(self.state, ckpt_name)
                elif type(self.config.saver.save_many) is list and curr_epoch in self.config.saver.save_many:
                    ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}_e{curr_epoch}.pth'
                    torch.save(self.state, ckpt_name)

            end = time.time()


def main():
    parser = argparse.ArgumentParser(description='ssl solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument("--tcp_port", type=str, default="5671")

    args = parser.parse_args()

    dist_init(port=str(args.tcp_port))
    # build solver
    solver = ImageNetSolver(args.config)

    if solver.config.data.last_iter < solver.config.data.max_iter:
        solver.train()
    else:
        solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
