import math

from .scheduler import _LRScheduler


class _LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        - optimizer (Optimizer): Wrapped optimizer.
        - lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        - last_iter (:obj:`int`): the index of last iteration.

    """

    def __init__(self, optimizer, lr_lambda, last_iter=0):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(_LambdaLR, self).__init__(optimizer, last_iter)

    def _get_new_lr(self):
        return [base_lr * lmbda(self.last_iter) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


class WCosConLR(_LambdaLR):
    """Cosine LR for group1, Constant LR for group2

    Args:
        - optimizer (:obj:`Optimizer`): Wrapped optimizer.
        - max_iter (:obj:`int`): maximum/total interrations of training.
        - base_lr (:obj:`float`): initial learning rate.
        - min_lr (:obj:`float`): minimum learning rate. Default: 0.
        - last_iter (:obj:`int`): the index of last iteration.

    """

    def __init__(self, optimizer, max_iter, warmup_steps, init_lr, base_lr, min_lr, last_iter=0):
        def lambda_network(curr_iter):
            if curr_iter <= warmup_steps:
                target_lr = init_lr + (base_lr - init_lr) * (curr_iter-1) / (warmup_steps-1)
            else:
                target_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (curr_iter - warmup_steps) / (max_iter - warmup_steps)))
            return target_lr / base_lr

        def lambda_pred_head(_):
            return 1.0

        super(WCosConLR, self).__init__(optimizer, [lambda_network, lambda_pred_head], last_iter)


WCosCon = WCosConLR
