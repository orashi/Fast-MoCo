import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD, based on
    `"Large Batch Training of Convolutional Networks" <https://arxiv.org/abs/1708.03888>`_

    Arguments:
        - params (:obj:`iterable`): iterable of parameters to optimize or dicts defining parameter groups
        - lr (:obj:`float`): learning rate
        - momentum (:obj:`float`, optional): momentum factor (default: 0)
        - weight_decay (:obj:`float`, optional): weight decay (L2 penalty) (default: 0)
        - dampening (:obj:`float`, optional): dampening for momentum (default: 0)
        - eta(:obj:`float`): LARS coefficient (default 0.001)
        - nesterov (:obj:`bool`, optional): enables Nesterov momentum (default: False)
        - eps(:obj:`float`, optional): epsilon (default 1e-8)
        - implementation (:obj:`string`, optional): 'PyTorch' or 'Sutskever' (default: PyTorch)


    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, eta=0.001, nesterov=False, eps=1e-8, implementation='PyTorch'):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        if implementation not in ['PyTorch', 'Sutskever']:
            raise ValueError(f"Invalid implementation option: {implementation}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, eta=eta, nesterov=nesterov, eps=eps, implementation=implementation)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            - closure (:obj:`callable`, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            eta = group['eta']
            eps = group['eps']
            implementation = group['implementation']
            lars_exclude = group['lars_exclude'] if 'lars_exclude' in group else False
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # compute local learning rate
                weight_norm = p.norm()
                grad_norm = d_p.norm()

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                    grad_norm = grad_norm.add(weight_norm, alpha=weight_decay)
                local_lr = eta * weight_norm / (grad_norm + eps) if not lars_exclude and weight_norm != 0 and grad_norm != 0 else 1.

                if implementation == 'Sutskever':
                    d_p = d_p.mul(local_lr * group['lr'])

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if implementation == 'PyTorch':
                    p.add_(d_p, alpha=-group['lr']*local_lr)
                elif implementation == 'Sutskever':
                    p.add_(d_p, alpha=-1.)

        return loss
