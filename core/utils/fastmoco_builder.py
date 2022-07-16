import torch
import torch.nn as nn
from core.utils import dist as link
from core.utils.misc import get_bn

from itertools import combinations


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super(projection_MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = BN(hidden_dim)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = BN(hidden_dim, affine=True)

        self.linear3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = BN(out_dim, affine=True)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.bn3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super(prediction_MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = BN(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        # layer 1
        x = self.linear1(input)
        x = self.bn1(x)
        hidden = self.activation(x)
        # N C
        x = self.layer2(hidden)
        return x


class FastMoCo(nn.Module):
    def __init__(self, backbone, bn, projector=None, predictor=None, ema=False, m=0.99, backbone_ema=None, arch='comb_patch',
                 split_num=2, combs=0):
        super(FastMoCo, self).__init__()
        projector = {} if projector is None else projector
        predictor = {} if predictor is None else predictor
        global BN

        BN = get_bn(bn)

        self.world_size = link.get_world_size()
        self.rank = link.get_rank()

        self.split_num = split_num
        self.m = m
        self.ema = ema
        self.arch = arch

        self.combs = combs

        self.dim_fc = dim_fc = backbone.fc.weight.shape[1]

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        _projector = projection_MLP(in_dim=dim_fc, **projector)
        self.encoder = nn.Sequential(
            self.backbone,
            _projector
        )

        if self.ema:
            self.bbone_ema = nn.Sequential(*list(backbone_ema.children())[:-1])
            _projector_ema = projection_MLP(in_dim=dim_fc, **projector)
            self.encoder_ema = nn.Sequential(
                self.bbone_ema,
                _projector_ema
            )

        self.predictor = prediction_MLP(**predictor)

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            param_t.data = param_t.data.mul_(self.m).add_(param_q.data, alpha=1. - self.m)

    def _local_split(self, x):     # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x

    def forward(self, input):
        x1, x2 = torch.split(input, [3, 3], dim=1)
        f, h = self.encoder, self.predictor

        if self.arch == 'comb_patch':
            x1_in_form = self._local_split(x1)
            x2_in_form = self._local_split(x2)

            z1_pre = f[0](x1_in_form)
            z2_pre = f[0](x2_in_form)

            z1_splits = list(z1_pre.split(z1_pre.size(0) // self.split_num ** 2, dim=0))  # 4b x c x
            z2_splits = list(z2_pre.split(z2_pre.size(0) // self.split_num ** 2, dim=0))

            z1_orthmix = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z1_splits, r=self.combs)))), dim=0) # 6 of 2combs / 4 of 3combs
            z2_orthmix = torch.cat(list(map(lambda x: sum(x) / self.combs, list(combinations(z2_splits, r=self.combs)))), dim=0) # 6 of 2combs / 4 of 3combs

            z1 = f[1](z1_orthmix)
            z2 = f[1](z2_orthmix)
        else:
            raise NotImplementedError
        # --> NxC, NxC

        p1, p2 = h(z1), h(z2)    # predictor

        if self.ema:
            with torch.no_grad():
                self._momentum_update_target_encoder()
                z1, z2 = self.encoder_ema(x1), self.encoder_ema(x2)
        else:
            with torch.no_grad():
                z1, z2 = f(x1), f(x2)

        return p1, z1, p2, z2
