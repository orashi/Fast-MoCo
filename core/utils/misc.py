import os
import logging
import torch
from core.utils import dist as link
import torch.distributed as dist
import numpy as np
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError:
    print('Import metrics failed!')

import yaml
from easydict import EasyDict

_logger = None
_logger_fh = None
_logger_names = []


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        dist.all_reduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


def makedir(path):
    if link.get_rank() == 0 and not os.path.exists(path):
        os.makedirs(path)
    dist.barrier()


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


class RankFilter(logging.Filter):
    def filter(self, record):
        return False


def create_logger(log_file, level=logging.INFO):
    global _logger, _logger_fh
    if _logger is None:
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        _logger.setLevel(level)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
        _logger_fh = fh
    else:
        _logger.removeHandler(_logger_fh)
        _logger.setLevel(level)

    return _logger


def get_logger(name, level=logging.INFO):
    global _logger_names
    logger = logging.getLogger(name)
    if name in _logger_names:
        return logger

    _logger_names.append(name)
    if link.get_rank() > 0:
        logger.addFilter(RankFilter())

    return logger


def reset_parameters(self) -> None:
    self.reset_running_stats()
    if self.affine:
        self.weight.data.uniform_()  # *default config used in SenseTime's internal SyncBatchNorm implementation, which
        self.bias.data.zero_()       # *leads to a different loss curve in first 15k iters (compared to pytorch default)
torch.nn.BatchNorm2d.reset_parameters = reset_parameters


def get_bn(config):
    if config.use_sync_bn:
        raise NotImplementedError("please use torch.nn.BatchNorm2d with conversion")
    else:
        def BNFunc(*args, **kwargs):
            return torch.nn.BatchNorm2d(*args, **kwargs, **config.kwargs)
        return BNFunc


def count_params(model):
    logger = get_logger(__name__)

    total = sum(p.numel() for p in model.parameters())
    conv = 0
    fc = 0
    others = 0
    for name, m in model.named_modules():
        # skip non-leaf modules
        if len(list(m.children())) > 0:
            continue
        num = sum(p.numel() for p in m.parameters())
        if isinstance(m, torch.nn.Conv2d):
            conv += num
        elif isinstance(m, torch.nn.Linear):
            fc += num
        else:
            others += num

    M = 1e6

    logger.info('total param: {:.3f}M, conv: {:.3f}M, fc: {:.3f}M, others: {:.3f}M'
                .format(total/M, conv/M, fc/M, others/M))


def count_flops(model, input_shape):
    logger = get_logger(__name__)

    flops_dict = {}

    def make_conv2d_hook(name):

        def conv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                / m.stride[0] / m.stride[1] / m.groups
            flops_dict[name] = int(flops)

        return conv2d_hook

    def make_fc_hook(name):

        def fc_hook(m, input):
            prod = 1
            for dim in input[0].size()[1:]:  # exclude batch size
                prod *= dim
            flops = prod * m.out_features
            flops_dict[name] = int(flops)
        return fc_hook

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            h = m.register_forward_pre_hook(make_conv2d_hook(name))
            hooks.append(h)
        elif isinstance(m, torch.nn.Linear):
            h = m.register_forward_pre_hook(make_fc_hook(name))
            hooks.append(h)

    input = torch.zeros(*input_shape).cuda()

    model.eval()
    with torch.no_grad():
        _ = model(input)

    model.train()
    total_flops = 0
    for k, v in flops_dict.items():
        # logger.info('module {}: {}'.format(k, v))
        total_flops += v
    logger.info('total FLOPS: {:.2f}M'.format(total_flops/1e6))

    for h in hooks:
        h.remove()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def detailed_metrics(output, target):
    precision_class = precision_score(target, output, average=None)
    recall_class = recall_score(target, output, average=None)
    f1_class = f1_score(target, output, average=None)
    precision_avg = precision_score(target, output, average='micro')
    recall_avg = recall_score(target, output, average='micro')
    f1_avg = f1_score(target, output, average='micro')
    return precision_class, recall_class, f1_class, precision_avg, recall_avg, f1_avg


def load_state_model(model, state):

    logger = get_logger(__name__)
    logger.info('======= loading model state... =======')

    model.load_state_dict(state, strict=False)

    state_keys = set(state.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - state_keys
    for k in missing_keys:
        logger.warn(f'missing key: {k}')


def load_state_optimizer(optimizer, state):

    logger = get_logger(__name__)
    logger.info('======= loading optimizer state... =======')

    optimizer.load_state_dict(state)


def modify_state(state, config):
    if 'state_dict' in state.keys() and 'model' not in state.keys():
        state['model'] = state.pop('state_dict')

    if hasattr(config, 'key'):
        for key in config['key']:
            if key == 'optimizer':
                state.pop(key)
            elif key == 'last_iter':
                state['last_iter'] = 0
            elif key == 'ema':
                state.pop('ema')
            else:
                state.pop(key)

    if hasattr(config, 'model'):
        for module in config['model']:
            state['model'].pop(module)

    if hasattr(config, 'model_match_start'):
        remove_list = []
        for k in state['model'].keys():
            for p in config['model_match_start']:
                if k.startswith(p):
                    remove_list.append(k)
                    break
        for k in remove_list:
            state['model'].pop(k)

    return state


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=0.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()

    target_a = target
    target_b = target[rand_index]

    # generate mixed sample
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank.T)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
