import torch
import torch.nn.functional as F
import torch.distributed as dist

from core.utils import dist as link

from torch.nn.modules.loss import _Loss


class InfoNCE(_Loss):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    @staticmethod
    def cosine_similarity(p, z):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        # [N E] [N E] -> [N] -> [1]
        return (p * z).sum(dim=1).mean()  # dot product & batch coeff normalization

    def loss(self, p, z_gather):
        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels)

    def forward(self, p1, z1, p2, z2):
        p1 = p1.split(z2.size(0), dim=0)
        p2 = p2.split(z1.size(0), dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        loss = 0
        for p in p1:
            loss = loss + self.loss(p, z2_gather)
        for p in p2:
            loss = loss + self.loss(p, z1_gather)

        return loss / (len(p1) + len(p2))


@torch.no_grad()
def concat_all_gather(tensor):
    """gather the given tensor"""
    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    dist.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0, async_op=False)
    return output
