import torch
import torch.nn.functional as F
import torch.distributed as dist
from core.utils import dist as link

from einops import rearrange, repeat
from torch.nn.modules.loss import _Loss


class InfoNCE(_Loss):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    @staticmethod
    def cosine_similarity(p, z):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        # [N E] [N E] -> [N] -> [1]
        return (p * z).sum(dim=1).mean()  # dot product & batch coeff normalization

    def loss(self, p, z):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)
        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels)

    def loss_reuse(self, p, z_gather):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels)

    def forward(self, p1, z1, p2, z2):
        return 0.5 * (self.loss(p1, z2.detach()) + self.loss(p2, z1.detach()))

    def forward_vis(self, p1, z1, p2, z2):
        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)

    def forward_vis_quad(self, p1, z1, p2, z2):
        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (self.weight * (infonce_loss_map + infonce_loss_indi) +
                       (1 - self.weight) * (cross_i_to_m + cross_m_to_i))

        return loss, ((infonce_loss_map + infonce_loss_indi) * 0.5, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_DT(InfoNCE):
    def forward_vis(self, p1, z1, p2, z2):
        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.25 * (infonce_loss_map + infonce_loss_indi + cross_m_to_i + cross_i_to_m)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_IOP(InfoNCE):
    def __init__(self, temperature, io_weight=1.):
        super(InfoNCE_IOP, self).__init__(temperature)
        self.io_weight = io_weight

    def loss_IOP(self, p, op_tags):
        return F.cross_entropy(p, op_tags)

    def forward_vis(self, p1, z1, op1, p2, z2, op2, op_tags):
        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        iop_loss = 0.5 * (self.loss_IOP(op1, op_tags[:, 0]) + self.loss_IOP(op2, op_tags[:, 1]))

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi) + iop_loss * self.io_weight

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, iop_loss)


class InfoNCE_IOP_LWContrast(InfoNCE_IOP):
    def forward_vis(self, p1, z1, op1, p2, z2, op2, op_tags):
        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        iop_loss_l1 = 0.5 * (self.loss_IOP(op1[0], op_tags[:, 0]) + self.loss_IOP(op2[0], op_tags[:, 1]))
        iop_loss_l2 = 0.5 * (self.loss_IOP(op1[1], op_tags[:, 0]) + self.loss_IOP(op2[1], op_tags[:, 1]))
        iop_loss_l3 = 0.5 * (self.loss_IOP(op1[2], op_tags[:, 0]) + self.loss_IOP(op2[2], op_tags[:, 1]))
        iop_loss_l4 = 0.5 * (self.loss_IOP(op1[3], op_tags[:, 0]) + self.loss_IOP(op2[3], op_tags[:, 1]))
        iop_loss_l5 = 0.5 * (self.loss_IOP(op1[4], op_tags[:, 0]) + self.loss_IOP(op2[4], op_tags[:, 1]))
        iop_loss = iop_loss_l1 + iop_loss_l2 + iop_loss_l3 + iop_loss_l4 + iop_loss_l5

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi) + iop_loss * self.io_weight

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, iop_loss,
                      iop_loss_l1, iop_loss_l2, iop_loss_l3, iop_loss_l4, iop_loss_l5)


class InfoNCE_IOP_LWContrast_detach(InfoNCE_IOP_LWContrast):
    def forward_vis(self, p1, z1, op1, p2, z2, op2, op_tags):
        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        iop_loss_l1 = 0.5 * (self.loss_IOP(op1[0], op_tags[:, 0]) + self.loss_IOP(op2[0], op_tags[:, 1]))
        iop_loss_l2 = 0.5 * (self.loss_IOP(op1[1], op_tags[:, 0]) + self.loss_IOP(op2[1], op_tags[:, 1]))
        iop_loss_l3 = 0.5 * (self.loss_IOP(op1[2], op_tags[:, 0]) + self.loss_IOP(op2[2], op_tags[:, 1]))
        iop_loss_l4 = 0.5 * (self.loss_IOP(op1[3], op_tags[:, 0]) + self.loss_IOP(op2[3], op_tags[:, 1]))
        iop_loss_l5 = 0.5 * (self.loss_IOP(op1[4], op_tags[:, 0]) + self.loss_IOP(op2[4], op_tags[:, 1]))
        iop_loss = iop_loss_l1 + iop_loss_l2 + iop_loss_l3 + iop_loss_l4 + iop_loss_l5

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = [0.5 * (infonce_loss_map + infonce_loss_indi), iop_loss * self.io_weight]

        return loss, (loss[0] + loss[1], infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, iop_loss,
                      iop_loss_l1, iop_loss_l2, iop_loss_l3, iop_loss_l4, iop_loss_l5)


class InfoNCE_IOP_LWContrast_2Stage(InfoNCE_IOP):
    def forward_vis(self, p1, z1, op1, p2, z2, op2, op_tags):
        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        iop_loss_l2 = 0.5 * (self.loss_IOP(op1[0], op_tags[:, 0]) + self.loss_IOP(op2[0], op_tags[:, 1]))
        iop_loss_l3 = 0.5 * (self.loss_IOP(op1[1], op_tags[:, 0]) + self.loss_IOP(op2[1], op_tags[:, 1]))
        iop_loss_l4 = 0.5 * (self.loss_IOP(op1[2], op_tags[:, 0]) + self.loss_IOP(op2[2], op_tags[:, 1]))
        iop_loss_l5 = 0.5 * (self.loss_IOP(op1[3], op_tags[:, 0]) + self.loss_IOP(op2[3], op_tags[:, 1]))
        iop_loss = iop_loss_l2 + iop_loss_l3 + iop_loss_l4 + iop_loss_l5

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi) + iop_loss * self.io_weight

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, iop_loss,
                      iop_loss_l2, iop_loss_l3, iop_loss_l4, iop_loss_l5)


class InfoNCE_halvlsp_wopool_sep(InfoNCE):
    def forward_vis(self, p1, z1, p2, z2):
        total_num = p1.size(0) // p2.size(0)
        p1 = p1.split(p2.size(0), dim=0)

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss(p, z2.detach())
            cross_m_to_i = cross_m_to_i + self.loss(p, z1.detach())
        infonce_loss_map = infonce_loss_map / total_num
        cross_m_to_i = cross_m_to_i / total_num

        infonce_loss_indi = self.loss(p2, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_ilsp_XNid(InfoNCE):
    def __init__(self, temperature, sp_level):
        super(InfoNCE_ilsp_XNid, self).__init__(temperature)
        self.sp_level = sp_level

    def forward_vis(self, ps, zt):
        ps = ps.split(ps.size(0) // self.sp_level, dim=0)

        zt = zt / zt.norm(dim=-1, keepdim=True)

        zt_gather = concat_all_gather(zt.detach())

        infonce_loss_map = 0
        for p in ps:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, zt_gather)
        infonce_loss_map = infonce_loss_map / self.sp_level

        return infonce_loss_map


class InfoNCE_ilsp_XNid_DO(InfoNCE):
    def __init__(self, temperature, sp_level):
        super(InfoNCE_ilsp_XNid_DO, self).__init__(temperature)
        self.sp_level = sp_level
        assert self.sp_level % 2 == 0

    def forward_vis(self, ps, zt):
        ps = ps.split(ps.size(0) // self.sp_level, dim=0)
        z1, z2 = zt.split(zt.size(0) // 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        for p in ps[:self.sp_level//2]:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
        for p in ps[self.sp_level//2:]:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z1_gather)
        infonce_loss_map = infonce_loss_map / self.sp_level

        return infonce_loss_map


class InfoNCE_ilsp_XNid_DO_MC(InfoNCE_ilsp_XNid_DO):
    def __init__(self, temperature, sp_level, i_base):
        super(InfoNCE_ilsp_XNid_DO_MC, self).__init__(temperature, sp_level)
        self.i_base = i_base
    def forward_vis(self, ps, p1, z1, p2, z2):
        ps = ps.split(ps.size(0) // self.i_base, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        for p in ps:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather) + self.loss_reuse(p, z1_gather)
        infonce_loss_map = infonce_loss_map + self.loss_reuse(p1, z2_gather) + self.loss_reuse(p2, z1_gather)
        infonce_loss_map = infonce_loss_map / self.sp_level

        return infonce_loss_map


class InfoNCE_ilsp_XNid_DO_DT(InfoNCE_ilsp_XNid_DO):
    def forward_vis(self, ps, zt):
        ps = ps.split(ps.size(0) // self.sp_level, dim=0)
        z1, z2 = zt.split(zt.size(0) // 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        for p in ps:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather) + self.loss_reuse(p, z1_gather)
        infonce_loss_map = infonce_loss_map / (self.sp_level * 2)

        return infonce_loss_map


class InfoNCE_ilsp_STRIPE(InfoNCE):
    def __init__(self, temperature, stripe_base):
        super(InfoNCE_ilsp_STRIPE, self).__init__(temperature)
        self.sp_level = stripe_base

    def forward_vis(self, ph, pw, zt):
        ph = ph.split(ph.size(0) // self.sp_level, dim=0)
        pw = pw.split(pw.size(0) // self.sp_level, dim=0)

        zt = zt / zt.norm(dim=-1, keepdim=True)

        zt_gather = concat_all_gather(zt.detach())

        infonce_loss_h = 0
        infonce_loss_w = 0
        for p in ph:
            infonce_loss_h = infonce_loss_h + self.loss_reuse(p, zt_gather)
        for p in pw:
            infonce_loss_w = infonce_loss_w + self.loss_reuse(p, zt_gather)
        infonce_loss_h = infonce_loss_h / self.sp_level
        infonce_loss_w = infonce_loss_w / self.sp_level

        loss = (infonce_loss_h + infonce_loss_w) * 0.5

        return loss, (infonce_loss_h, infonce_loss_w)


class InfoNCE_ilsp_MIXNid(InfoNCE):
    def __init__(self, i_base, temperature, res_base):
        super(InfoNCE_ilsp_MIXNid, self).__init__(temperature)
        self.sp_level = res_base
        self.i_base = i_base
        self.i_portion = i_base - 1

    def forward_vis(self, ps, pb, zt):
        ps = ps.split(ps.size(0) // self.sp_level, dim=0)
        pb = pb.split(pb.size(0) // 2, dim=0)

        zt = zt / zt.norm(dim=-1, keepdim=True)

        zt_gather, zt_p_gather, zt_o_gather = concat_all_gather_isplit(zt.detach(), self.i_base, self.i_portion)

        infonce_loss_map = 0
        infonce_loss_indi = 0
        for p in ps:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, torch.cat([zt_p_gather, zt_o_gather], dim=0))
        for p in pb:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, torch.cat([zt_o_gather, zt_p_gather], dim=0))
        infonce_loss_map = infonce_loss_map / self.sp_level
        infonce_loss_indi = infonce_loss_indi / 2

        loss = infonce_loss_map / self.i_base * self.i_portion + infonce_loss_indi / self.i_base

        return loss, (loss, infonce_loss_map, infonce_loss_indi)


class InfoNCE_halvlsp_post_sep(InfoNCE):
    def __init__(self, temperature, lsp_level, sep_weight=0.5):
        super(InfoNCE_halvlsp_post_sep, self).__init__(temperature)
        self.lsp_level = lsp_level
        self.sep_weight = sep_weight

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_halvlsp_selfneg_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())
        z2_hard = torch.cat([z2_gather, z1_gather], dim=0)

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_hard)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_31lsp_post_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z2 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1 = z1[z1.size(0) // 2:]

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_31lsp_fix_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:z1.size(0) // 2]
        z1_o = z1[z1.size(0) // 2:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:z2.size(0)], z2_gather) * 2
                                                   + self.loss_reuse(p[z2.size(0):], torch.cat([z1_p_gather, z1_o_gather], dim=0))
                                                   ) / 3

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, torch.cat([z1_o_gather, z1_p_gather], dim=0))
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_11lsp_symmetric_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0)//2)*1]
        z1_o = z1[(z1.size(0)//2)*1:]
        z2_p = z2[:(z2.size(0)//2)*1]
        z2_o = z2[(z2.size(0)//2)*1:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_p_gather = concat_all_gather(z2_p.detach())
        z2_o_gather = concat_all_gather(z2_o.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_31lsp_symmetric_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0)//4)*3]
        z1_o = z1[(z1.size(0)//4)*3:]
        z2_p = z2[:(z2.size(0)//4)*3]
        z2_o = z2[(z2.size(0)//4)*3:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_p_gather = concat_all_gather(z2_p.detach())
        z2_o_gather = concat_all_gather(z2_o.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_71lsp_symmetric_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0)//8)*7]
        z1_o = z1[(z1.size(0)//8)*7:]
        z2_p = z2[:(z2.size(0)//8)*7]
        z2_o = z2[(z2.size(0)//8)*7:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_p_gather = concat_all_gather(z2_p.detach())
        z2_o_gather = concat_all_gather(z2_o.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_71lsp_symmetric_sep_threshold(InfoNCE_71lsp_symmetric_sep):
    def __init__(self, threshold_rate, temperature, lsp_level, sep_weight=0.5):
        super(InfoNCE_71lsp_symmetric_sep_threshold, self).__init__(temperature, lsp_level, sep_weight)
        self.threshold_rate = threshold_rate

    def loss_threshold(self, p1, z1_p_gather, z1_o_gather, z2_p_gather, z2_o_gather):

        infonce_loss_map = []
        for p in p1:
            infonce_loss_map = infonce_loss_map + [self.loss_t_logits(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0)),
                                                   self.loss_t_logits(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))]
        infonce_loss_map = torch.cat(infonce_loss_map, dim=0)

        loss = - infonce_loss_map.topk(int(infonce_loss_map.size(0) * self.threshold_rate))[0].mean()

        return loss

    def loss_t_logits(self, p, z_gather):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)].diag()  # [N_local]

        return p_z_probs

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0)//8)*7]
        z1_o = z1[(z1.size(0)//8)*7:]
        z2_p = z2[:(z2.size(0)//8)*7]
        z2_o = z2[(z2.size(0)//8)*7:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_p_gather = concat_all_gather(z2_p.detach())
        z2_o_gather = concat_all_gather(z2_o.detach())


        infonce_loss_map = self.loss_threshold(p1, z1_p_gather, z1_o_gather, z2_p_gather, z2_o_gather)
        cross_m_to_i = torch.tensor(0)

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)



class InfoNCE_71lsp_symmetric_sep_threshold_clip(InfoNCE_71lsp_symmetric_sep_threshold):
    def loss_threshold(self, p1, z1_p_gather, z1_o_gather, z2_p_gather, z2_o_gather):

        infonce_loss_map = []
        for p in p1:
            infonce_loss_map = infonce_loss_map + [self.loss_t_logits(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0)),
                                                   self.loss_t_logits(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))]
        infonce_loss_map = torch.cat(infonce_loss_map, dim=0)

        in_thres = infonce_loss_map.topk(int(infonce_loss_map.size(0) * self.threshold_rate))[0]
        in_thres_threshold = in_thres[-1]
        out_thres = infonce_loss_map.topk(int(infonce_loss_map.size(0) * (1 - self.threshold_rate)), largest=False)[0]
        out_thres = out_thres * (in_thres_threshold / out_thres).detach()
        infonce_loss_map = torch.cat([in_thres, out_thres], dim=0)
        loss = - infonce_loss_map.mean()

        return loss


class InfoNCE_71lsp_symmetric_ppl_sep(InfoNCE_71lsp_symmetric_sep):
    def __init__(self, temperature, lsp_level, w_isp=0.8, w_avg=0.2, sep_weight=0.5):
        super(InfoNCE_71lsp_symmetric_ppl_sep, self).__init__(temperature, lsp_level, sep_weight)
        self.w_isp = w_isp
        self.w_avg = w_avg

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p1_sum = sum(p1) / self.lsp_level ** 2
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0)//8)*7]
        z1_o = z1[(z1.size(0)//8)*7:]
        z2_p = z2[:(z2.size(0)//8)*7]
        z2_o = z2[(z2.size(0)//8)*7:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_p_gather = concat_all_gather(z2_p.detach())
        z2_o_gather = concat_all_gather(z2_o.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss_postpre = (self.loss_reuse(p1_sum[:p1_sum.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0)) +
                        self.loss_reuse(p1_sum[p1_sum.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) * 0.5

        loss = (infonce_loss_map * self.w_isp + loss_postpre * self.w_avg) * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class CosSim_71lsp_symmetric_sep(InfoNCE_71lsp_symmetric_sep):
    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)

        z1_p = z1[:(z1.size(0)//8)*7]
        z1_o = z1[(z1.size(0)//8)*7:]
        z2_p = z2[:(z2.size(0)//8)*7]
        z2_o = z2[(z2.size(0)//8)*7:]

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.cosine_similarity_loss(p[:p.size(0)//2], z2_p.detach())
                                                   + self.cosine_similarity_loss(p[p.size(0)//2:], z1_p.detach())) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.cosine_similarity_loss(p2[:p2.size(0)//2], z2_o.detach()) + self.cosine_similarity_loss(p2[p2.size(0)//2:], z1_o.detach())) / 2
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_151lsp_symmetric_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0)//16)*15]
        z1_o = z1[(z1.size(0)//16)*15:]
        z2_p = z2[:(z2.size(0)//16)*15]
        z2_o = z2[(z2.size(0)//16)*15:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_p_gather = concat_all_gather(z2_p.detach())
        z2_o_gather = concat_all_gather(z2_o.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_311lsp_symmetric_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0)//32)*31]
        z1_o = z1[(z1.size(0)//32)*31:]
        z2_p = z2[:(z2.size(0)//32)*31]
        z2_o = z2[(z2.size(0)//32)*31:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_p_gather = concat_all_gather(z2_p.detach())
        z2_o_gather = concat_all_gather(z2_o.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_71lsp_fix_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0) // 4) * 3]
        z1_o = z1[(z1.size(0) // 4) * 3:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:z2.size(0)], z2_gather) * 2
                                                   + self.loss_reuse(p[z2.size(0):], torch.cat([z1_p_gather, z1_o_gather], dim=0))
                                                   ) / 3

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, torch.cat([z1_o_gather, z1_p_gather], dim=0))
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_151lsp_fix_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0) // 8) * 7]
        z1_o = z1[(z1.size(0) // 8) * 7:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:z2.size(0)], z2_gather) * 2
                                                   + self.loss_reuse(p[z2.size(0):], torch.cat([z1_p_gather, z1_o_gather], dim=0))
                                                   ) / 3

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, torch.cat([z1_o_gather, z1_p_gather], dim=0))
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_311lsp_fix_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        # z2_v1 = torch.cat([z2, z1[:z1.size(0) // 2]], dim=0)
        z1_p = z1[:(z1.size(0) // 16) * 15]
        z1_o = z1[(z1.size(0) // 16) * 15:]

        z1_p_gather = concat_all_gather(z1_p.detach())
        z1_o_gather = concat_all_gather(z1_o.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:z2.size(0)], z2_gather) * 2
                                                   + self.loss_reuse(p[z2.size(0):], torch.cat([z1_p_gather, z1_o_gather], dim=0))
                                                   ) / 3

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, torch.cat([z1_o_gather, z1_p_gather], dim=0))
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_71lsp_post_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z2 = torch.cat([z2, z1[:z1.size(0) // 4]], dim=0)
        z1 = z1[z1.size(0) // 4:]

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = torch.tensor(0)

        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_ilsp_OnT_postpool(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_ilsp_OnT(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = sum(z1.split(z1.size(0) // self.lsp_level ** 2, dim=0)) / self.lsp_level ** 2
        z2 = sum(z2.split(z2.size(0) // self.lsp_level ** 2, dim=0)) / self.lsp_level ** 2

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_stripe(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level
        cross_m_to_i = cross_m_to_i / self.lsp_level

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level
        cross_i_to_m = cross_i_to_m / self.lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)

class InfoNCE_cb4(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        infonce_loss_map = infonce_loss_map + self.loss_reuse(p1, z2_gather)
        cross_m_to_i = cross_m_to_i + self.loss_reuse(p1, z1_gather)


        infonce_loss_indi = 0
        cross_i_to_m = 0

        infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p2, z1_gather)
        cross_i_to_m = cross_i_to_m + self.loss_reuse(p2, z2_gather)

        infonce_loss_indi = infonce_loss_indi
        cross_i_to_m = cross_i_to_m

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)

class InfoNCE_ilsp_post_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_ilsp_post_sep_norm(InfoNCE_ilsp_post_sep):
    def __init__(self, l1, temperature, lsp_level, sep_weight=0.5):
        super(InfoNCE_ilsp_post_sep_norm, self).__init__(temperature, lsp_level, sep_weight)
        self.l1_weight = l1
    def forward_vis(self, p1, z1, p2, z2):
        lasso = (torch.mean(p1.abs() / p1.norm(dim=-1, keepdim=True),dim=0).sum() + torch.mean(p2.abs() / p2.norm(dim=-1, keepdim=True),dim=0).sum())/2 * self.l1_weight

        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2


        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + lasso

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_ilsp_sep_plus_prepred_seppred(InfoNCE_halvlsp_post_sep):
    def __init__(self, temperature, lsp_level, sep_weight=0.5, w_isp=0.8, w_avg=0.2):
        super(InfoNCE_ilsp_sep_plus_prepred_seppred, self).__init__(temperature, lsp_level, sep_weight)
        self.w_isp = w_isp
        self.w_avg = w_avg

    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = loss * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_sep_plus_combs(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level
        cross_m_to_i = cross_m_to_i / self.lsp_level

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level
        cross_i_to_m = cross_i_to_m / self.lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = loss * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_sep_plus_combs_nn(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, K, **kwargs):
        super(InfoNCE_ilsp_sep_plus_combs_nn, self).__init__(**kwargs)
        self.K = K

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z1_gather, queue_1, z2_gather, queue_2, queue_ptr):
        batch_size = z1_gather.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue_1[:, ptr:ptr + batch_size] = z1_gather.T
        queue_2[:, ptr:ptr + batch_size] = z2_gather.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _nn(self, z_gather, queue):
        sims = z_gather.mm(queue)   #[B, K]
        nn_idx = sims.argmax(dim=1)
        return torch.index_select(queue, dim=1, index=nn_idx)  #

    def loss_reuse(self, p, z_gather):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels)

    def forward_vis(self, p1, z1, p2, z2, queue_1, queue_2, queue_ptr):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        # nearest neighbor replacement
        z1_nn = self._nn(z1_gather, queue_1)
        z2_nn = self._nn(z2_gather, queue_2)
        #

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_nn)
        infonce_loss_map = infonce_loss_map / self.lsp_level

        infonce_loss_indi = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_nn)
        infonce_loss_indi = infonce_loss_indi / self.lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        self._dequeue_and_enqueue(z1_gather, queue_1, z2_gather, queue_2, queue_ptr)

        return loss, (loss, infonce_loss_map, infonce_loss_indi)


class InfoNCE_ilsp_sep_plus_combs_self(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z1_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level
        cross_m_to_i = cross_m_to_i / self.lsp_level

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z2_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level
        cross_i_to_m = cross_i_to_m / self.lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = loss * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_sep_plus_combs_clean(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level

        infonce_loss_indi = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss


class InfoNCE_ilsp_sep_plus_combs_localloss(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, ll_weight, **kwargs):
        super(InfoNCE_ilsp_sep_plus_combs_localloss, self).__init__(**kwargs)
        self.ll_weight = ll_weight

    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, p2, z2,
                    z1_map, z2_map, z1_t_map, z2_t_map,
                    p1_m, p2_m, z1_t_map_proj, z2_t_map_proj):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level

        infonce_loss_indi = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level

        loss_ori = (infonce_loss_map + infonce_loss_indi) * 0.5

        ####
        p1_m = rearrange(p1_m, '(b h w) c -> b c h w', b=z1_map.size(0), h=z1_map.size(2), w=z1_map.size(3))
        p2_m = rearrange(p2_m, '(b h w) c -> b c h w', b=z2_map.size(0), h=z2_map.size(2), w=z2_map.size(3))

        z1_map = torch.stack(z1_map.split(z1_map.size(0) // 4, dim=0), dim=2) # (b c p h w)
        z2_map = torch.stack(z2_map.split(z2_map.size(0) // 4, dim=0), dim=2)
        p1_m = torch.stack(p1_m.split(p1_m.size(0) // 4, dim=0), dim=2) # (b c p h w)
        p2_m = torch.stack(p2_m.split(p2_m.size(0) // 4, dim=0), dim=2)


        z1_map = rearrange(z1_map, 'b c p h w -> b (p h w) c')
        z2_map = rearrange(z2_map, 'b c p h w -> b (p h w) c')
        p1_m = rearrange(p1_m, 'b c p h w -> (b p h w) c')
        p2_m = rearrange(p2_m, 'b c p h w -> (b p h w) c')

        z1_t_map_proj = rearrange(z1_t_map_proj, '(b h w) c -> b (h w) c', b=z1_t_map.size(0), h=z1_t_map.size(2), w=z1_t_map.size(3))
        z2_t_map_proj = rearrange(z2_t_map_proj, '(b h w) c -> b (h w) c', b=z2_t_map.size(0), h=z2_t_map.size(2), w=z2_t_map.size(3))

        z1_t_map = rearrange(z1_t_map, 'b c h w -> b (h w) c')
        z2_t_map = rearrange(z2_t_map, 'b c h w -> b (h w) c')

        idx_1 = torch.matmul(F.normalize(z1_map, p=2, dim=-1),
                                           F.normalize(z2_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[1]  # B x (p h w) x (h w) -> B x (p h w)
        idx_2 = torch.matmul(F.normalize(z2_map, p=2, dim=-1),
                                           F.normalize(z1_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[1]  # B x (p h w) x (h w) -> B x (p h w)


        t1_indexed_region = torch.gather(z2_t_map_proj, 1, idx_1.unsqueeze(2).expand(-1, -1, z2_t_map_proj.size(2)))  # B x (p h w) x c (index matrix: B, T_s, 1)
        t2_indexed_region = torch.gather(z1_t_map_proj, 1, idx_2.unsqueeze(2).expand(-1, -1, z1_t_map_proj.size(2)))  # B x (p h w) x c (index matrix: B, T_s, 1)

        indexed_region_1 = rearrange(t1_indexed_region, 'b s c -> (b s) c')
        indexed_region_2 = rearrange(t2_indexed_region, 'b s c -> (b s) c')

        loss_ll = (self.cosine_similarity_loss(p1_m, indexed_region_1.detach()) + self.cosine_similarity_loss(p2_m, indexed_region_2.detach())) * 0.5

        loss = loss_ori * (1 - self.ll_weight) + loss_ll * self.ll_weight

        return loss, (loss, loss_ori, loss_ll)


class InfoNCE_ilsp_sep_plus_combs_localloss_simloss(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, ll_weight, **kwargs):
        super(InfoNCE_ilsp_sep_plus_combs_localloss_simloss, self).__init__(**kwargs)
        self.ll_weight = ll_weight

    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, p2, z2,
                    z1_map, z2_map, z1_t_map, z2_t_map,
                    p1_m, p2_m, z1_t_map_proj, z2_t_map_proj):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.cosine_similarity_loss(p, z2.detach())

        infonce_loss_map = infonce_loss_map / self.lsp_level

        infonce_loss_indi = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.cosine_similarity_loss(p, z1.detach())

        infonce_loss_indi = infonce_loss_indi / self.lsp_level

        loss_ori = (infonce_loss_map + infonce_loss_indi) * 0.5

        ####
        p1_m = rearrange(p1_m, '(b h w) c -> b c h w', b=z1_map.size(0), h=z1_map.size(2), w=z1_map.size(3))
        p2_m = rearrange(p2_m, '(b h w) c -> b c h w', b=z2_map.size(0), h=z2_map.size(2), w=z2_map.size(3))

        z1_map = torch.stack(z1_map.split(z1_map.size(0) // 4, dim=0), dim=2)  # (b c p h w)
        z2_map = torch.stack(z2_map.split(z2_map.size(0) // 4, dim=0), dim=2)
        p1_m = torch.stack(p1_m.split(p1_m.size(0) // 4, dim=0), dim=2)  # (b c p h w)
        p2_m = torch.stack(p2_m.split(p2_m.size(0) // 4, dim=0), dim=2)

        z1_map = rearrange(z1_map, 'b c p h w -> b (p h w) c')
        z2_map = rearrange(z2_map, 'b c p h w -> b (p h w) c')
        p1_m = rearrange(p1_m, 'b c p h w -> (b p h w) c')
        p2_m = rearrange(p2_m, 'b c p h w -> (b p h w) c')

        z1_t_map_proj = rearrange(z1_t_map_proj, '(b h w) c -> b (h w) c', b=z1_t_map.size(0), h=z1_t_map.size(2),
                                  w=z1_t_map.size(3))
        z2_t_map_proj = rearrange(z2_t_map_proj, '(b h w) c -> b (h w) c', b=z2_t_map.size(0), h=z2_t_map.size(2),
                                  w=z2_t_map.size(3))

        z1_t_map = rearrange(z1_t_map, 'b c h w -> b (h w) c')
        z2_t_map = rearrange(z2_t_map, 'b c h w -> b (h w) c')

        idx_1 = torch.matmul(F.normalize(z1_map, p=2, dim=-1),
                             F.normalize(z2_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[
            1]  # B x (p h w) x (h w) -> B x (p h w)
        idx_2 = torch.matmul(F.normalize(z2_map, p=2, dim=-1),
                             F.normalize(z1_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[
            1]  # B x (p h w) x (h w) -> B x (p h w)

        t1_indexed_region = torch.gather(z2_t_map_proj, 1, idx_1.unsqueeze(2).expand(-1, -1, z2_t_map_proj.size(
            2)))  # B x (p h w) x c (index matrix: B, T_s, 1)
        t2_indexed_region = torch.gather(z1_t_map_proj, 1, idx_2.unsqueeze(2).expand(-1, -1, z1_t_map_proj.size(
            2)))  # B x (p h w) x c (index matrix: B, T_s, 1)

        indexed_region_1 = rearrange(t1_indexed_region, 'b s c -> (b s) c')
        indexed_region_2 = rearrange(t2_indexed_region, 'b s c -> (b s) c')

        loss_ll = (self.cosine_similarity_loss(p1_m, indexed_region_1.detach()) + self.cosine_similarity_loss(p2_m,
                                                                                                              indexed_region_2.detach())) * 0.5

        loss = loss_ori * (1 - self.ll_weight) + loss_ll * self.ll_weight

        return loss, (loss, loss_ori, loss_ll)


class InfoNCE_ilsp_sep_plus_combs_localloss_simloss_test(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, ll_weight, **kwargs):
        super(InfoNCE_ilsp_sep_plus_combs_localloss_simloss_test, self).__init__(**kwargs)
        self.ll_weight = ll_weight

    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, p2, z2,
                    z1_map, z2_map, z1_t_map, z2_t_map,
                    p1_m, p2_m, z1_t_map_proj, z2_t_map_proj):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.cosine_similarity_loss(p, z2.detach())

        infonce_loss_map = infonce_loss_map / self.lsp_level

        infonce_loss_indi = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.cosine_similarity_loss(p, z1.detach())

        infonce_loss_indi = infonce_loss_indi / self.lsp_level

        loss_ori = (infonce_loss_map + infonce_loss_indi) * 0.5

        ####
        p1_m = rearrange(p1_m, '(b h w) c -> b c h w', b=z1_map.size(0), h=z1_map.size(2), w=z1_map.size(3))
        p2_m = rearrange(p2_m, '(b h w) c -> b c h w', b=z2_map.size(0), h=z2_map.size(2), w=z2_map.size(3))

        z1_map = torch.stack(z1_map.split(z1_map.size(0) // 4, dim=0), dim=2)  # (b c p h w)
        z2_map = torch.stack(z2_map.split(z2_map.size(0) // 4, dim=0), dim=2)
        p1_m = torch.stack(p1_m.split(p1_m.size(0) // 4, dim=0), dim=2)  # (b c p h w)
        p2_m = torch.stack(p2_m.split(p2_m.size(0) // 4, dim=0), dim=2)

        _b, _c, _p, _h, _w = z1_map.shape
        z1_map = rearrange(z1_map, 'b c p h w -> b (p h w) c')
        z2_map = rearrange(z2_map, 'b c p h w -> b (p h w) c')
        p1_m = rearrange(p1_m, 'b c p h w -> (b p h w) c')
        p2_m = rearrange(p2_m, 'b c p h w -> (b p h w) c')

        z1_t_map_proj = rearrange(z1_t_map_proj, '(b h w) c -> b (h w) c', b=z1_t_map.size(0), h=z1_t_map.size(2),
                                  w=z1_t_map.size(3))
        z2_t_map_proj = rearrange(z2_t_map_proj, '(b h w) c -> b (h w) c', b=z2_t_map.size(0), h=z2_t_map.size(2),
                                  w=z2_t_map.size(3))

        z1_t_map = rearrange(z1_t_map, 'b c h w -> b (h w) c')
        z2_t_map = rearrange(z2_t_map, 'b c h w -> b (h w) c')

        idx_1 = torch.matmul(F.normalize(z1_map, p=2, dim=-1),
                             F.normalize(z2_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[
            1]  # B x (p h w) x (h w) -> B x (p h w)
        idx_2 = torch.matmul(F.normalize(z2_map, p=2, dim=-1),
                             F.normalize(z1_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[
            1]  # B x (p h w) x (h w) -> B x (p h w)

        t1_indexed_region = torch.gather(z2_t_map_proj, 1, idx_1.unsqueeze(2).expand(-1, -1, z2_t_map_proj.size(
            2)))  # B x (p h w) x c (index matrix: B, T_s, 1)
        t2_indexed_region = torch.gather(z1_t_map_proj, 1, idx_2.unsqueeze(2).expand(-1, -1, z1_t_map_proj.size(
            2)))  # B x (p h w) x c (index matrix: B, T_s, 1)

        indexed_region_1 = rearrange(t1_indexed_region, 'b (p h w) c -> (b p h w) c', p=_p, h=_h, w=_w)
        indexed_region_2 = rearrange(t2_indexed_region, 'b (p h w) c -> (b p h w) c', p=_p, h=_h, w=_w)

        loss_ll = (self.cosine_similarity_loss(p1_m, indexed_region_1.detach()) + self.cosine_similarity_loss(p2_m,
                                                                                                              indexed_region_2.detach())) * 0.5

        loss = loss_ori * (1 - self.ll_weight) + loss_ll * self.ll_weight

        return loss, (loss, loss_ori, loss_ll)


class InfoNCE_ilsp_sep_plus_baseline_localloss_simloss(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, ll_weight, **kwargs):
        super(InfoNCE_ilsp_sep_plus_baseline_localloss_simloss, self).__init__(**kwargs)
        self.ll_weight = ll_weight

    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, p2, z2,
                    z1_map, z2_map, z1_t_map, z2_t_map,
                    p1_m, p2_m, z1_t_map_proj, z2_t_map_proj):

        infonce_loss_map = self.cosine_similarity_loss(p1, z2.detach())
        infonce_loss_indi = self.cosine_similarity_loss(p2, z1.detach())

        loss_ori = (infonce_loss_map + infonce_loss_indi) * 0.5

        ####
        p1_m = rearrange(p1_m, '(b h w) c -> b c h w', b=z1_map.size(0), h=z1_map.size(2), w=z1_map.size(3))
        p2_m = rearrange(p2_m, '(b h w) c -> b c h w', b=z2_map.size(0), h=z2_map.size(2), w=z2_map.size(3))

        z1_map = rearrange(z1_map, 'b c h w -> b (h w) c')
        z2_map = rearrange(z2_map, 'b c h w -> b (h w) c')
        p1_m = rearrange(p1_m, 'b c h w -> (b h w) c')
        p2_m = rearrange(p2_m, 'b c h w -> (b h w) c')

        z1_t_map_proj = rearrange(z1_t_map_proj, '(b h w) c -> b (h w) c', b=z1_t_map.size(0), h=z1_t_map.size(2),
                                  w=z1_t_map.size(3))
        z2_t_map_proj = rearrange(z2_t_map_proj, '(b h w) c -> b (h w) c', b=z2_t_map.size(0), h=z2_t_map.size(2),
                                  w=z2_t_map.size(3))

        z1_t_map = rearrange(z1_t_map, 'b c h w -> b (h w) c')
        z2_t_map = rearrange(z2_t_map, 'b c h w -> b (h w) c')

        idx_1 = torch.matmul(F.normalize(z1_map, p=2, dim=-1),
                             F.normalize(z2_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[
            1]  # B x (p h w) x (h w) -> B x (p h w)
        idx_2 = torch.matmul(F.normalize(z2_map, p=2, dim=-1),
                             F.normalize(z1_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[
            1]  # B x (p h w) x (h w) -> B x (p h w)

        t1_indexed_region = torch.gather(z2_t_map_proj, 1, idx_1.unsqueeze(2).expand(-1, -1, z2_t_map_proj.size(
            2)))  # B x (p h w) x c (index matrix: B, T_s, 1)
        t2_indexed_region = torch.gather(z1_t_map_proj, 1, idx_2.unsqueeze(2).expand(-1, -1, z1_t_map_proj.size(
            2)))  # B x (p h w) x c (index matrix: B, T_s, 1)

        indexed_region_1 = rearrange(t1_indexed_region, 'b s c -> (b s) c')
        indexed_region_2 = rearrange(t2_indexed_region, 'b s c -> (b s) c')

        loss_ll = (self.cosine_similarity_loss(p1_m, indexed_region_1.detach()) + self.cosine_similarity_loss(p2_m,
                                                                                                              indexed_region_2.detach())) * 0.5

        loss = loss_ori * (1 - self.ll_weight) + loss_ll * self.ll_weight

        return loss, (loss, loss_ori, loss_ll)


class InfoNCE_ilsp_sep_plus_baseline_localloss(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, ll_weight, **kwargs):
        super(InfoNCE_ilsp_sep_plus_baseline_localloss, self).__init__(**kwargs)
        self.ll_weight = ll_weight

    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, p2, z2,
                    z1_map, z2_map, z1_t_map, z2_t_map,
                    p1_m, p2_m, z1_t_map_proj, z2_t_map_proj):
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = self.loss_reuse(p1, z2_gather)
        infonce_loss_indi = self.loss_reuse(p2, z1_gather)

        loss_ori = (infonce_loss_map + infonce_loss_indi) * 0.5

        ####
        p1_m = rearrange(p1_m, '(b h w) c -> b c h w', b=z1_map.size(0), h=z1_map.size(2), w=z1_map.size(3))
        p2_m = rearrange(p2_m, '(b h w) c -> b c h w', b=z2_map.size(0), h=z2_map.size(2), w=z2_map.size(3))

        z1_map = rearrange(z1_map, 'b c h w -> b (h w) c')
        z2_map = rearrange(z2_map, 'b c h w -> b (h w) c')
        p1_m = rearrange(p1_m, 'b c h w -> (b h w) c')
        p2_m = rearrange(p2_m, 'b c h w -> (b h w) c')

        z1_t_map_proj = rearrange(z1_t_map_proj, '(b h w) c -> b (h w) c', b=z1_t_map.size(0), h=z1_t_map.size(2), w=z1_t_map.size(3))
        z2_t_map_proj = rearrange(z2_t_map_proj, '(b h w) c -> b (h w) c', b=z2_t_map.size(0), h=z2_t_map.size(2), w=z2_t_map.size(3))

        z1_t_map = rearrange(z1_t_map, 'b c h w -> b (h w) c')
        z2_t_map = rearrange(z2_t_map, 'b c h w -> b (h w) c')

        idx_1 = torch.matmul(F.normalize(z1_map, p=2, dim=-1),
                                           F.normalize(z2_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[1]  # B x (p h w) x (h w) -> B x (p h w)
        idx_2 = torch.matmul(F.normalize(z2_map, p=2, dim=-1),
                                           F.normalize(z1_t_map, p=2, dim=-1).permute(0, 2, 1)).max(dim=2)[1]  # B x (p h w) x (h w) -> B x (p h w)


        t1_indexed_region = torch.gather(z2_t_map_proj, 1, idx_1.unsqueeze(2).expand(-1, -1, z2_t_map_proj.size(2)))  # B x (p h w) x c (index matrix: B, T_s, 1)
        t2_indexed_region = torch.gather(z1_t_map_proj, 1, idx_2.unsqueeze(2).expand(-1, -1, z1_t_map_proj.size(2)))  # B x (p h w) x c (index matrix: B, T_s, 1)

        indexed_region_1 = rearrange(t1_indexed_region, 'b s c -> (b s) c')
        indexed_region_2 = rearrange(t2_indexed_region, 'b s c -> (b s) c')

        loss_ll = (self.cosine_similarity_loss(p1_m, indexed_region_1.detach()) + self.cosine_similarity_loss(p2_m, indexed_region_2.detach())) * 0.5

        loss = loss_ori * (1 - self.ll_weight) + loss_ll * self.ll_weight

        return loss, (loss, loss_ori, loss_ll)


class InfoNCE_ilsp_sep_plus_combsSE_smooth(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, smooth_ratio=0.1, **kwargs):
        super(InfoNCE_ilsp_sep_plus_combsSE_smooth, self).__init__(**kwargs)
        self.smooth_ratio = smooth_ratio
        target_ratio = 1 - smooth_ratio * 3
        self.ratios = [smooth_ratio, target_ratio, smooth_ratio, smooth_ratio]

    def forward_vis_base(self, ps, z1_gather, z2_gather):
        p1, p2 = ps
        lsp_level = len(p1)

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)

        infonce_loss_map = infonce_loss_map / lsp_level

        infonce_loss_indi = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)

        infonce_loss_indi = infonce_loss_indi / lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss

    def split(self, p):
        p1, p2, p3, p4 = p
        bsize = p4.size(0)
        return (p1.split(bsize, dim=0), p2.split(bsize, dim=0), p3.split(bsize, dim=0), [p4]) # c41 ... c44

    def forward_vis(self, p1, z1, p2, z2):
        p1 = self.split(p1)
        p2 = self.split(p2)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        losses = []
        for _p in zip(p1, p2):
            losses.append(self.forward_vis_base(_p, z1_gather, z2_gather))

        loss = sum([_loss * self.ratios[i] for i, _loss in enumerate(losses)])

        return loss, (loss, *losses)


class InfoNCE_ilsp_sep_plus_combsSE_reweight(InfoNCE_ilsp_sep_plus_combsSE_smooth):
    def loss_reuse(self, p, z_gather):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels, reduction='none')

    def forward_vis(self, p1, z1, p2, z2):
        p1 = self.split(p1)
        p2 = self.split(p2)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        losses = []
        for _p in zip(p1, p2):
            losses.append(self.forward_vis_base(_p, z1_gather, z2_gather))

        losses[1] = losses[1] * losses[1].mean().detach() / losses[1].detach()

        loss = sum([_loss.mean() * self.ratios[i] for i, _loss in enumerate(losses)])

        return loss, (loss, *[_loss.mean() for _loss in losses])


class InfoNCE_ilsp_sep_plus_combsSE_rebalance(InfoNCE_ilsp_sep_plus_combsSE_reweight):
    def __init__(self, type='c44', **kwargs):
        super(InfoNCE_ilsp_sep_plus_combsSE_rebalance, self).__init__(**kwargs)
        self.type = type

    def forward_vis(self, p1, z1, p2, z2):
        p1 = self.split(p1)
        p2 = self.split(p2)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        losses = []
        for _p in zip(p1, p2):
            losses.append(self.forward_vis_base(_p, z1_gather, z2_gather))

        loss_anchor = losses[1].mean()
        # losses[1] = losses[1] * losses[1].mean().detach() / losses[1].detach()
        if self.type == 'c44':
            loss_c41_mask = losses[0] < loss_anchor
            loss_c44_mask = losses[3] > loss_anchor
            loss_c42_mask = (losses[0] >= loss_anchor) & (losses[3] <= loss_anchor)
            loss = (losses[0] * loss_c41_mask + losses[1] * loss_c42_mask + losses[3] * loss_c44_mask).mean()
        elif self.type == 'c43':
            loss_c41_mask = losses[0] < loss_anchor
            loss_c43_mask = losses[2] > loss_anchor
            loss_c42_mask = (losses[0] >= loss_anchor) & (losses[2] <= loss_anchor)
            loss = (losses[0] * loss_c41_mask + losses[1] * loss_c42_mask + losses[2] * loss_c43_mask).mean()
        else: raise

        return loss, (loss, *[_loss.mean() for _loss in losses])



class InfoNCE_ilsp_sep_plus_combsSE_rand(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def loss_reuse(self, p, z_gather, pidx):
        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()[pidx]
        p_z_m = p[pidx].mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels)

    def forward_vis_base(self, p1, z1_gather, p2, z2_gather, pidx):
        lsp_level = len(p1)

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather, pidx)

        infonce_loss_map = infonce_loss_map / lsp_level

        infonce_loss_indi = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather, pidx)

        infonce_loss_indi = infonce_loss_indi / lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss

    def forward_vis(self, p1, z1, p2, z2):
        p1_1, p1_2, p1_3, p1_4 = p1
        p2_1, p2_2, p2_3, p2_4 = p2

        self.bsize = p1_4.size(0)

        p1_4 = [p1_4]
        p2_4 = [p2_4]
        p1_2 = p1_2.split(self.bsize, dim=0)
        p2_2 = p2_2.split(self.bsize, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        # self.cosine_similarity(z1, z2).topk(self.top)[1]

        loss_4 = self.forward_vis_base(p1_4, z1_gather,
                                       p2_4, z2_gather,
                                       torch.arange(0, self.bsize // 4))

        loss_2 = self.forward_vis_base(p1_2, z1_gather,
                                       p2_2, z2_gather,
                                       torch.arange(self.bsize // 4, self.bsize))

        loss = loss_4 / 4 + loss_2 / 4 * 3

        return loss, (loss, loss_4, loss_2)


class InfoNCE_ilsp_sep_plus_combsSE_top(InfoNCE_ilsp_sep_plus_combsSE_rand):
    def __init__(self, easiest_first=False, **kwargs):
        super(InfoNCE_ilsp_sep_plus_combsSE_top, self).__init__(**kwargs)
        self.easiest_first = easiest_first
    @staticmethod
    def cosine_similarity_keep(p, z):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        # [N E] [N E] -> [N] -> [1]
        return (p * z).sum(dim=1)  # dot product & batch coeff normalization

    def forward_vis(self, p1, z1, p2, z2):
        p1_1, p1_2, p1_3, p1_4 = p1
        p2_1, p2_2, p2_3, p2_4 = p2

        self.bsize = p1_4.size(0)

        p1_4 = [p1_4]
        p2_4 = [p2_4]
        p1_2 = p1_2.split(self.bsize, dim=0)
        p2_2 = p2_2.split(self.bsize, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)


        idx = self.cosine_similarity_keep(z1, z2).topk(self.bsize, largest=self.easiest_first)[1]

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())


        loss_4 = self.forward_vis_base(p1_4, z1_gather,
                                       p2_4, z2_gather,
                                       idx[:self.bsize // 4])

        loss_2 = self.forward_vis_base(p1_2, z1_gather,
                                       p2_2, z2_gather,
                                       idx[self.bsize // 4:])

        loss = loss_4 / 4 + loss_2 / 4 * 3

        return loss, (loss, loss_4, loss_2)


class InfoNCE_ilsp_sep_plus_combsSE_sync(InfoNCE_ilsp_sep_plus_combsSE_top):
    def __init__(self, alpha=0.25, **kwargs):
        super(InfoNCE_ilsp_sep_plus_combsSE_sync, self).__init__(**kwargs)
        self.alpha = alpha
    def forward_vis(self, p1, z1, p2, z2):
        p1_1, p1_2, p1_3, p1_4 = p1
        p2_1, p2_2, p2_3, p2_4 = p2

        self.bsize = p1_4.size(0)

        p1_4 = [p1_4]
        p2_4 = [p2_4]
        p1_2 = p1_2.split(self.bsize, dim=0)
        p2_2 = p2_2.split(self.bsize, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        idx = self.cosine_similarity_keep(z1_gather, z2_gather).topk(self.bsize * link.get_world_size(), largest=self.easiest_first)[1]

        partition = self.bsize * link.get_world_size() * self.alpha
        assert partition % 1 == 0

        offset = link.get_rank() * self.bsize
        c4_idx = idx[:int(partition)]
        c4_idx_local = c4_idx[(c4_idx >= offset) & (c4_idx < offset + self.bsize)] - offset
        mask = torch.ones(self.bsize, dtype=torch.bool).cuda()
        mask[c4_idx_local] = False
        c2_idx_local = torch.arange(0, self.bsize, dtype=torch.long).cuda()[mask]

        if len(c4_idx_local) != 0:
            loss_4 = self.forward_vis_base(p1_4, z1_gather,
                                           p2_4, z2_gather,
                                           c4_idx_local)
        else:
            loss_4 = self.forward_vis_base(p1_4, z1_gather,
                                           p2_4, z2_gather,
                                           torch.tensor([0])) * 0 # dummy

        if len(c2_idx_local) != 0:
            loss_2 = self.forward_vis_base(p1_2, z1_gather,
                                           p2_2, z2_gather,
                                           c2_idx_local)
        else:
            loss_2 = self.forward_vis_base(p1_2, z1_gather,
                                           p2_2, z2_gather,
                                           torch.tensor([0])) * 0 # dummy

        c4_weight_local = len(c4_idx_local) / self.bsize
        c2_weight_local = len(c2_idx_local) / self.bsize

        loss = loss_4 * c4_weight_local + loss_2 * c2_weight_local

        return loss, (loss, loss_4 * c4_weight_local / self.alpha, loss_2 * c2_weight_local / (1 - self.alpha))


class InfoNCE_ilsp_sep_plus_combs_mc(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2, p1_2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level
        cross_m_to_i = cross_m_to_i / self.lsp_level

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level
        cross_i_to_m = cross_i_to_m / self.lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = (loss * 2 * self.lsp_level + self.loss_reuse(p1_2, z2_gather) * self.w_isp) / (2 * self.lsp_level + self.w_isp)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_sep_plus_combs_mc_strict(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, p2, z2, p1_2):
        p1_2 = p1_2.split(p1_2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1_2:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level
        cross_m_to_i = cross_m_to_i / self.lsp_level

        loss_indi = (self.loss_reuse(p1, z2_gather) + self.loss_reuse(p2, z1_gather)) / 2

        loss = (loss_indi * 2 + infonce_loss_map * self.lsp_level * 2) / (self.lsp_level * 2 + 2)

        return loss, (loss, infonce_loss_map, loss_indi, cross_m_to_i)


class InfoNCE_ilsp_sep_plus_combs_mc_strict_balance(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, p2, z2, p1_2):
        p1_2 = p1_2.split(p1_2.size(0) // self.lsp_level, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1_2:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level
        cross_m_to_i = cross_m_to_i / self.lsp_level

        loss_indi = (self.loss_reuse(p1, z2_gather) + self.loss_reuse(p2, z1_gather)) / 2

        loss = (loss_indi * 2 + infonce_loss_map * self.lsp_level) / (self.lsp_level + 2)

        return loss, (loss, infonce_loss_map, loss_indi, cross_m_to_i)


class CosSim_InfoNCE_ilsp_sep_plus_combs(InfoNCE_ilsp_sep_plus_combs):
    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level, dim=0)

        p1_sum = pp1
        p2_sum = pp2

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.cosine_similarity_loss(p, z2.detach())
            cross_m_to_i = cross_m_to_i + self.cosine_similarity_loss(p, z1.detach())

        infonce_loss_map = infonce_loss_map / self.lsp_level
        cross_m_to_i = cross_m_to_i / self.lsp_level

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.cosine_similarity_loss(p, z1.detach())
            cross_i_to_m = cross_i_to_m + self.cosine_similarity_loss(p, z2.detach())

        infonce_loss_indi = infonce_loss_indi / self.lsp_level
        cross_i_to_m = cross_i_to_m / self.lsp_level

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.cosine_similarity_loss(p1_sum, z2.detach()) + self.cosine_similarity_loss(p2_sum, z1.detach())) * 0.5
        loss = loss * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_8Nid_combs(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, ps, z3):
        ps = ps.split(ps.size(0) // self.lsp_level, dim=0)

        z3 = z3 / z3.norm(dim=-1, keepdim=True)

        z3_gather = concat_all_gather(z3.detach())

        infonce_loss_map = 0
        for p in ps:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z3_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level

        loss = infonce_loss_map

        return loss, (loss, infonce_loss_map)



class InfoNCE_ilsp_sep_plus_prepred_frec(InfoNCE_halvlsp_post_sep):
    def __init__(self, temperature, lsp_level, sep_weight=0.5, w_isp=0.8, w_avg=0.2, w_frec=1):
        super(InfoNCE_ilsp_sep_plus_prepred_frec, self).__init__(temperature, lsp_level, sep_weight)
        self.w_isp = w_isp
        self.w_avg = w_avg
        self.w_frec = w_frec

    def forward_vis(self, p1, z1, pp1, p2, z2, pp2, z1_frec, z2_frec, z1_ft, z2_ft):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        z_frec = torch.cat([z1_frec, z2_frec], dim=0)
        z_ft = torch.cat([z1_ft, z2_ft], dim=0)

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss_frec_mse = F.mse_loss(z_frec, torch.flatten(z_ft, 1), reduction='mean')
        loss = loss * self.w_isp + loss_postpre * self.w_avg + loss_frec_mse * self.w_frec

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre, loss_frec_mse)



class InfoNCE_ilsp_sep_ppl_recs(InfoNCE_halvlsp_post_sep):
    def __init__(self, temperature, lsp_level, sep_weight=0.5, w_isp=0.8, w_avg=0.2, w_rec=0.1):
        super(InfoNCE_ilsp_sep_ppl_recs, self).__init__(temperature, lsp_level, sep_weight)
        self.w_isp = w_isp
        self.w_avg = w_avg
        self.w_rec = w_rec
        self.default_index = torch.tensor([[4, 3, 2],
                                           [1, 4, 3],
                                           [2, 1, 4],
                                           [3, 2, 1]])

    def forward_vis(self, p1, z1, pp1, p2, z2, pp2, recs, ozs):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        oz_1, oz_2 = ozs.split(ozs.size(0) // 2, dim=0)  # 2[4b x dim]
        oz_1, oz_2 = rearrange(oz_1, '(s b) d -> b s d', s=4), rearrange(oz_2, '(s b) d -> b s d', s=4)  # 2[b x 4 x dim]
        ozs = torch.cat([oz_1[:, i] for i in self.default_index] +
                        [oz_2[:, i] for i in self.default_index], dim=0)  # 8b x 3 x dim
        recon_loss = F.mse_loss(recs, ozs)

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = loss * self.w_isp + loss_postpre * self.w_avg + recon_loss * self.w_rec

        return loss, (recon_loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_pplNet(InfoNCE_halvlsp_post_sep):
    def __init__(self, temperature, w_1, w_2, w_4, lsp_level=2, sep_weight=0.5):
        super(InfoNCE_ilsp_pplNet, self).__init__(temperature, lsp_level, sep_weight)
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_4 = w_4

    def forward_vis(self, z1, z2, ps1_1, ps1_2, ps2_1, ps2_2, ps4_1, ps4_2): #p1, z1, pp1, p2, z2, pp2):
        p1 = ps1_1.split(ps1_1.size(0) // 4, dim=0)
        p2 = ps1_2.split(ps1_2.size(0) // 4, dim=0)

        p1_sum2 = ps2_1.split(ps1_1.size(0) // 4, dim=0)
        p2_sum2 = ps2_2.split(ps1_2.size(0) // 4, dim=0)

        p1_sum4 = ps4_1
        p2_sum4 = ps4_2

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        ###ps1
        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)
        for p in p2:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z1_gather)
        infonce_loss_map = infonce_loss_map / 8
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2
        ###ps1
        ###ps2
        infonce_ppl2 = 0
        for p in p1_sum2:
            infonce_ppl2 = infonce_ppl2 + self.loss_reuse(p, z2_gather)
        for p in p2_sum2:
            infonce_ppl2 = infonce_ppl2 + self.loss_reuse(p, z1_gather)
        infonce_ppl2 = infonce_ppl2 / 4
        ###ps2
        infonce_ppl4 = (self.loss_reuse(p1_sum4, z2_gather) + self.loss_reuse(p2_sum4, z1_gather)) / 2

        loss = infonce_loss_map * self.w_1 + infonce_ppl2 * self.w_2 + infonce_ppl4 * self.w_4

        return loss, (infonce_loss_map, infonce_ppl2, infonce_ppl4, cross_m_to_i)


class InfoNCE_ilsp_8Nid_4ppl_seppred(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, p2, z2, z3, pp1, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z3 = z3 / z3.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z3_gather = concat_all_gather(z3.detach())

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z3_gather)
        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_map2 = 0
        for p in p2:
            infonce_loss_map2 = infonce_loss_map2 + self.loss_reuse(p, z3_gather)
        infonce_loss_map2 = infonce_loss_map2 / self.lsp_level ** 2

        infonce_loss_map = (infonce_loss_map + infonce_loss_map2) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z3_gather) + self.loss_reuse(p2_sum, z3_gather)) * 0.5
        loss = infonce_loss_map * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, loss_postpre)


class InfoNCE_ilsp_8Nid_2ppl_seppred(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, p2, z2, z3, pps):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z3 = z3 / z3.norm(dim=-1, keepdim=True)

        z3_gather = concat_all_gather(z3.detach())

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z3_gather)
        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_map2 = 0
        for p in p2:
            infonce_loss_map2 = infonce_loss_map2 + self.loss_reuse(p, z3_gather)
        infonce_loss_map2 = infonce_loss_map2 / self.lsp_level ** 2

        infonce_loss_map = (infonce_loss_map + infonce_loss_map2) * 0.5

        loss_postpre = sum([self.loss_reuse(pp, z3_gather) for pp in pps]) / len(pps)
        loss = infonce_loss_map * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, loss_postpre)


class InfoNCE_ilsp_8Nid_seppred(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, p2, z2, z3, pp):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z3 = z3 / z3.norm(dim=-1, keepdim=True)

        z3_gather = concat_all_gather(z3.detach())

        infonce_loss_map = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z3_gather)
        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_map2 = 0
        for p in p2:
            infonce_loss_map2 = infonce_loss_map2 + self.loss_reuse(p, z3_gather)
        infonce_loss_map2 = infonce_loss_map2 / self.lsp_level ** 2

        infonce_loss_map = (infonce_loss_map + infonce_loss_map2) * 0.5

        loss_postpre = self.loss_reuse(pp, z3_gather)
        loss = infonce_loss_map * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, loss_postpre)


class InfoNCE_ilsp_8Nid_pplNet_seppred(InfoNCE_halvlsp_post_sep):
    def __init__(self, temperature, w_1, w_2, w_4, w_8, lsp_level=2, sep_weight=0.5):
        super(InfoNCE_ilsp_8Nid_pplNet_seppred, self).__init__(temperature, lsp_level, sep_weight)
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_4 = w_4
        self.w_8 = w_8

    def forward_vis(self, p1, p_avg2, p_avg4, p_avg8, z):
        p1 = p1.split(p1.size(0) // 8, dim=0)
        p_avg2 = p_avg2.split(p_avg2.size(0) // 4, dim=0)
        p_avg4 = p_avg4.split(p_avg4.size(0) // 2, dim=0)
        p_avg8 = p_avg8

        z = z / z.norm(dim=-1, keepdim=True)
        z_gather = concat_all_gather(z.detach())

        infonce_loss_map = sum([self.loss_reuse(p, z_gather) for p in p1]) / 8
        infonce_ppl2 = sum([self.loss_reuse(p, z_gather) for p in p_avg2]) / 4
        infonce_ppl4 = sum([self.loss_reuse(p, z_gather) for p in p_avg4]) / 2
        infonce_ppl8 = self.loss_reuse(p_avg8, z_gather)

        loss = (infonce_loss_map * self.w_1 +
                infonce_ppl2 * self.w_2 +
                infonce_ppl4 * self.w_4 +
                infonce_ppl8 * self.w_8)

        return loss, (loss, infonce_loss_map, infonce_ppl2, infonce_ppl4, infonce_ppl8)

class InfoNCE_ilsp_sep_plus_prepred_seppred_fullprojs(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, Tema, temperature, lsp_level, sep_weight=0.5, w_isp=1, w_emance=1, w_avg=0.5, Starget='selfe'):
        super(InfoNCE_ilsp_sep_plus_prepred_seppred_fullprojs, self).__init__(temperature, lsp_level, sep_weight, w_isp, w_avg)
        self.Tema = Tema
        self.Starget = Starget
        self.w_emance = w_emance

    def forward_vis(self, p1, z1, ep1, ez1, eze1, pp1, p2, z2, ep2, ez2, eze2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        p1_sum = pp1
        p2_sum = pp2

        if self.Tema:
            zz1 = eze1 / eze1.norm(dim=-1, keepdim=True)
            zz2 = eze2 / eze2.norm(dim=-1, keepdim=True)
        else:
            zz1 = ez1 / ez1.norm(dim=-1, keepdim=True)
            zz2 = ez2 / ez2.norm(dim=-1, keepdim=True)
        zz1_gather = concat_all_gather(zz1.detach())
        zz2_gather = concat_all_gather(zz2.detach())

        if self.Starget == 'selfe':
            z1 = z1 / z1.norm(dim=-1, keepdim=True)
            z2 = z2 / z2.norm(dim=-1, keepdim=True)

            z1_gather = concat_all_gather(z1.detach())
            z2_gather = concat_all_gather(z2.detach())
        elif self.Starget == 'emaz':
            if self.Tema:
                z1 = ez1 / ez1.norm(dim=-1, keepdim=True)
                z2 = ez2 / ez2.norm(dim=-1, keepdim=True)

                z1_gather = concat_all_gather(z1.detach())
                z2_gather = concat_all_gather(z2.detach())
            else:
                z1_gather = zz1_gather
                z2_gather = zz2_gather
        elif self.Starget == 'emazema':
            if self.Tema:
                z1_gather = zz1_gather
                z2_gather = zz2_gather
            else:
                z1 = eze1 / eze1.norm(dim=-1, keepdim=True)
                z2 = eze2 / eze2.norm(dim=-1, keepdim=True)

                z1_gather = concat_all_gather(z1.detach())
                z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        infonce_loss_z_back_map = (self.loss_reuse(ep1, zz2_gather) + self.loss_reuse(ep2, zz1_gather)) * 0.5
        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = loss * self.w_isp + loss_postpre * self.w_avg + infonce_loss_z_back_map * self.w_emance

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre, infonce_loss_z_back_map)


class InfoNCE_ilsp_sep_hardneg(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, z1_local, p2, z2, z2_local):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_local = z1_local / z1_local.norm(dim=-1, keepdim=True)
        z2_local = z2_local / z2_local.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        z1_local_gather = concat_all_gather_woself(z1_local.detach())
        z2_local_gather = concat_all_gather_woself(z2_local.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, torch.cat([z2_gather, z2_local_gather], dim=0))
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, torch.cat([z1_gather, z1_local_gather], dim=0))
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_ilsp_sep_NNsub(InfoNCE_halvlsp_post_sep):
    def loss_NNsub(self, p, z_gather):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        p_z_m = p.mm(z_gather.T) / self.temperature  # [N_local, N]
        labels = p_z_m.topk(1, dim=-1)[1].squeeze().cuda()

        return F.cross_entropy(p_z_m, labels)

    def forward_vis(self, p1, z1, z1_local, p2, z2, z2_local):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_local = z1_local / z1_local.norm(dim=-1, keepdim=True)
        z2_local = z2_local / z2_local.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        z1_local_gather = concat_all_gather_woself(z1_local.detach())
        z2_local_gather = concat_all_gather_woself(z2_local.detach())

        infonce_loss_map = 0
        infonce_loss_NNsub = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            infonce_loss_NNsub = infonce_loss_NNsub + self.loss_NNsub(p, z2_local_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)
            infonce_loss_NNsub = infonce_loss_NNsub + self.loss_NNsub(p, z1_local_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2
        infonce_loss_NNsub = infonce_loss_NNsub / (2 * self.lsp_level ** 2)

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_NNsub * 0.2

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_NNsub)


class InfoNCE_ilsp_sep_plus_prepred_dual(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = 0
        for i in range(len(p1_sum)):
            loss_postpre = loss_postpre + (self.loss_reuse(p1_sum[i], z2_gather) + self.loss_reuse(p2_sum[i], z1_gather)) * 0.5 / len(p1_sum)

        loss = loss * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_sep_plus_prepred_selfreg(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1
        p2_sum = pp2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z1_gather) + self.loss_reuse(p2_sum, z2_gather)) * 0.5
        loss = loss * self.w_isp + loss_postpre * self.w_avg

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_Nilsp_sep_plus_prepred_seppred(InfoNCE_ilsp_sep_plus_prepred_seppred):
    def __init__(self, i_base, temperature, lsp_level, w_isp=0.8, w_avg=0.2):
        super(InfoNCE_Nilsp_sep_plus_prepred_seppred, self).__init__(temperature, lsp_level, 1 - 1 / i_base, w_isp, w_avg)
        self.i_base = i_base
        self.i_portion = i_base - 1

    def forward_vis(self, p1, z1, pp1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1

        z1_gather, z1_p_gather, z1_o_gather = concat_all_gather_isplit(z1.detach(), self.i_base, self.i_portion)
        z2_gather, z2_p_gather, z2_o_gather = concat_all_gather_isplit(z2.detach(), self.i_base, self.i_portion)

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)



        loss_postpre = (self.loss_reuse(p1_sum[:p1_sum.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0)) +
                        self.loss_reuse(p1_sum[p1_sum.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) * 0.5

        loss = (infonce_loss_map * self.w_isp + loss_postpre * self.w_avg) * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_Nilsp_sep_plus_combs(InfoNCE_Nilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1

        z1_gather, z1_p_gather, z1_o_gather = concat_all_gather_isplit(z1.detach(), self.i_base, self.i_portion)
        z2_gather, z2_p_gather, z2_o_gather = concat_all_gather_isplit(z2.detach(), self.i_base, self.i_portion)

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)


        loss_postpre = (self.loss_reuse(p1_sum[:p1_sum.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0)) +
                        self.loss_reuse(p1_sum[p1_sum.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) * 0.5

        loss = (infonce_loss_map * self.w_isp + loss_postpre * self.w_avg) * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class CosSim_InfoNCE_Nilsp_sep_plus_combs(InfoNCE_Nilsp_sep_plus_combs):
    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, pp1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level, dim=0)

        p1_sum = pp1

        z1_p = z1[:(z1.size(0)//self.i_base)*self.i_portion]
        z1_o = z1[(z1.size(0)//self.i_base)*self.i_portion:]
        z2_p = z2[:(z2.size(0)//self.i_base)*self.i_portion]
        z2_o = z2[(z2.size(0)//self.i_base)*self.i_portion:]

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.cosine_similarity_loss(p[:p.size(0)//2], z2_p.detach()) +
                                                   self.cosine_similarity_loss(p[p.size(0)//2:], z1_p.detach())) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level

        infonce_loss_indi = (self.cosine_similarity_loss(p2[:p2.size(0)//2], z2_o.detach()) +
                             self.cosine_similarity_loss(p2[p2.size(0)//2:], z1_o.detach())) / 2
        cross_i_to_m = torch.tensor(0)


        loss_postpre = (self.cosine_similarity_loss(p1_sum[:p1_sum.size(0)//2], z2_p.detach()) +
                        self.cosine_similarity_loss(p1_sum[p1_sum.size(0)//2:], z1_p.detach())) * 0.5

        loss = (infonce_loss_map * self.w_isp + loss_postpre * self.w_avg) * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)



class CosSim_Nilsp_sep_plus_prepred_seppred(InfoNCE_Nilsp_sep_plus_prepred_seppred):
    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, pp1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)

        p1_sum = pp1

        z1_p = z1[:(z1.size(0)//self.i_base)*self.i_portion]
        z1_o = z1[(z1.size(0)//self.i_base)*self.i_portion:]
        z2_p = z2[:(z2.size(0)//self.i_base)*self.i_portion]
        z2_o = z2[(z2.size(0)//self.i_base)*self.i_portion:]

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.cosine_similarity_loss(p[:p.size(0)//2], z2_p.detach()) +
                                                   self.cosine_similarity_loss(p[p.size(0)//2:], z1_p.detach())) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.cosine_similarity_loss(p2[:p2.size(0)//2], z2_o.detach()) +
                             self.cosine_similarity_loss(p2[p2.size(0)//2:], z1_o.detach())) / 2
        cross_i_to_m = torch.tensor(0)

        loss_postpre = (self.cosine_similarity_loss(p1_sum[:p1_sum.size(0)//2], z2_p.detach()) +
                        self.cosine_similarity_loss(p1_sum[p1_sum.size(0)//2:], z1_p.detach())) * 0.5

        loss = (infonce_loss_map * self.w_isp + loss_postpre * self.w_avg) * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_Nilsp_sep_plus_septarget(InfoNCE_halvlsp_post_sep):
    def __init__(self, i_base, temperature, lsp_level, local_weight):
        super(InfoNCE_Nilsp_sep_plus_septarget, self).__init__(temperature, lsp_level, 1 - 1 / i_base)
        self.i_base = i_base
        self.i_portion = i_base - 1
        self.local_weight = local_weight

    def forward_vis(self, p1, z1, p2, z2, z1_local, z2_local):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather, z1_p_gather, z1_o_gather = concat_all_gather_isplit(z1.detach(), self.i_base, self.i_portion)
        z2_gather, z2_p_gather, z2_o_gather = concat_all_gather_isplit(z2.detach(), self.i_base, self.i_portion)
        local_z1_p_gathers, local_z1_o_gathers = concat_all_gather_4sp_isplit(z1_local.detach(), self.i_base, self.i_portion, self.lsp_level)
        local_z2_p_gathers, local_z2_o_gathers = concat_all_gather_4sp_isplit(z2_local.detach(), self.i_base, self.i_portion, self.lsp_level)

        # loss_map, loss_map_local
        infonce_loss_map = 0
        infonce_loss_map_local = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

            local_loss_map = []
            for i in range(self.lsp_level ** 2):
                local_loss_map.append((self.loss_reuse(p[:p.size(0)//2], torch.cat([local_z2_p_gathers[i], local_z2_o_gathers[i]], dim=0))
                                                       + self.loss_reuse(p[p.size(0)//2:], torch.cat([local_z1_p_gathers[i], local_z1_o_gathers[i]], dim=0))) / 2)
            local_loss_map = torch.stack(local_loss_map, dim=0)
            infonce_loss_map_local = infonce_loss_map_local + torch.mean(local_loss_map)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        infonce_loss_map_local = infonce_loss_map_local / self.lsp_level ** 2


        # loss_indi
        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2

        # loss_indi_local
        infonce_loss_indi_local = []
        for i in range(self.lsp_level ** 2):
            local_loss_indi = (self.loss_reuse(p2[:p2.size(0) // 2], torch.cat([local_z2_o_gathers[i], local_z2_p_gathers[i]], dim=0)) +
                               self.loss_reuse(p2[p2.size(0) // 2:], torch.cat([local_z1_o_gathers[i], local_z1_p_gathers[i]], dim=0))) / 2

            infonce_loss_indi_local.append(local_loss_indi)
        infonce_loss_indi_local = torch.mean(torch.stack(infonce_loss_indi_local, dim=0))

        loss = (infonce_loss_map*(1-self.local_weight) + infonce_loss_map_local*self.local_weight) * self.sep_weight + \
               (infonce_loss_indi*(1-self.local_weight) + infonce_loss_indi_local*self.local_weight) * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, infonce_loss_map_local, infonce_loss_indi_local)


class InfoNCE_Nilsp_sep_plus_septarget_seppred(InfoNCE_Nilsp_sep_plus_septarget):
    def forward_vis(self, p1, z1, p2, z2, z1_local, z2_local, pp1, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        pp1 = pp1.split(pp1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather, z1_p_gather, z1_o_gather = concat_all_gather_isplit(z1.detach(), self.i_base, self.i_portion)
        z2_gather, z2_p_gather, z2_o_gather = concat_all_gather_isplit(z2.detach(), self.i_base, self.i_portion)
        local_z1_p_gathers, local_z1_o_gathers = concat_all_gather_4sp_isplit(z1_local.detach(), self.i_base, self.i_portion, self.lsp_level)
        local_z2_p_gathers, local_z2_o_gathers = concat_all_gather_4sp_isplit(z2_local.detach(), self.i_base, self.i_portion, self.lsp_level)

        # loss_map, loss_map_local
        infonce_loss_map = 0
        infonce_loss_map_local = 0
        for p_i, p in enumerate(p1):
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2

            local_loss_map = []
            _p = pp1[p_i]
            for i in range(self.lsp_level ** 2):
                local_loss_map.append((self.loss_reuse(_p[:_p.size(0)//2], torch.cat([local_z2_p_gathers[i], local_z2_o_gathers[i]], dim=0))
                                                       + self.loss_reuse(_p[_p.size(0)//2:], torch.cat([local_z1_p_gathers[i], local_z1_o_gathers[i]], dim=0))) / 2)
            local_loss_map = torch.stack(local_loss_map, dim=0)
            infonce_loss_map_local = infonce_loss_map_local + torch.mean(local_loss_map)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        infonce_loss_map_local = infonce_loss_map_local / self.lsp_level ** 2


        # loss_indi
        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2

        # loss_indi_local
        infonce_loss_indi_local = []
        for i in range(self.lsp_level ** 2):
            local_loss_indi = (self.loss_reuse(pp2[:pp2.size(0) // 2], torch.cat([local_z2_o_gathers[i], local_z2_p_gathers[i]], dim=0)) +
                               self.loss_reuse(pp2[pp2.size(0) // 2:], torch.cat([local_z1_o_gathers[i], local_z1_p_gathers[i]], dim=0))) / 2

            infonce_loss_indi_local.append(local_loss_indi)
        infonce_loss_indi_local = torch.mean(torch.stack(infonce_loss_indi_local, dim=0))

        loss = (infonce_loss_map*(1-self.local_weight) + infonce_loss_map_local*self.local_weight) * self.sep_weight + \
               (infonce_loss_indi*(1-self.local_weight) + infonce_loss_indi_local*self.local_weight) * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, infonce_loss_map_local, infonce_loss_indi_local)


class InfoNCE_Nilsp_sep_plus_prepred_seppred_oneside(InfoNCE_Nilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1

        z1_gather, z1_p_gather, z1_o_gather = concat_all_gather_isplit(z1.detach(), self.i_base, self.i_portion)
        z2_gather, z2_p_gather, z2_o_gather = concat_all_gather_isplit(z2.detach(), self.i_base, self.i_portion)

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z2_p_gather, z2_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss_postpre = (self.loss_reuse(p1_sum[:p1_sum.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0)) +
                        self.loss_reuse(p1_sum[p1_sum.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) * 0.5

        loss = (infonce_loss_map * self.w_isp + loss_postpre * self.w_avg) * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_Nilsp_sep_plus_prepred_seppred_oneside(InfoNCE_Nilsp_sep_plus_prepred_seppred):
    def forward_vis(self, p1, z1, pp1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = pp1

        z1_gather, z1_p_gather, z1_o_gather = concat_all_gather_isplit(z1.detach(), self.i_base, self.i_portion)
        z2_gather, z2_p_gather, z2_o_gather = concat_all_gather_isplit(z2.detach(), self.i_base, self.i_portion)

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z2_p_gather, z2_o_gather], dim=0))) / 2

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        loss_postpre = (self.loss_reuse(p1_sum[:p1_sum.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0)) +
                        self.loss_reuse(p1_sum[p1_sum.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) * 0.5

        loss = (infonce_loss_map * self.w_isp + loss_postpre * self.w_avg) * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_post_sep_plus_postpre_seppred(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = sum(pp1.split(pp1.size(0) // self.lsp_level ** 2, dim=0)) / self.lsp_level ** 2
        p2_sum = sum(pp1.split(pp1.size(0) // self.lsp_level ** 2, dim=0)) / self.lsp_level ** 2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = loss * 0.8 + loss_postpre * 0.2

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_post_sep_plus_postpre(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        p1_sum = sum(p1) / self.lsp_level ** 2
        p2_sum = sum(p2) / self.lsp_level ** 2

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5

        loss_postpre = (self.loss_reuse(p1_sum, z2_gather) + self.loss_reuse(p2_sum, z1_gather)) * 0.5
        loss = loss * 0.8 + loss_postpre * 0.2

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, loss_postpre)


class InfoNCE_ilsp_zback_sep(InfoNCE_halvlsp_post_sep):
    def loss_reuse_full(self, p_gather, z_gather):
        # [N, E]

        p = p_gather / p_gather.norm(dim=-1, keepdim=True)

        labels = torch.arange(0, p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels)

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * z1.size(0)

        z1_gather = concat_all_gather(z1.detach())
        z1_gather_backable = torch.cat([z1_gather[:offset], z1, z1_gather[offset + z1.size(0):]], dim=0)
        z2_gather = concat_all_gather(z2.detach())
        z2_gather_backable = torch.cat([z2_gather[:offset], z2, z2_gather[offset + z2.size(0):]], dim=0)

        infonce_loss_z_back_map = 0

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)
            p_gather = concat_all_gather(p.detach())
            infonce_loss_z_back_map = infonce_loss_z_back_map + self.loss_reuse_full(p_gather, z2_gather_backable)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)
            p_gather = concat_all_gather(p.detach())
            infonce_loss_z_back_map = infonce_loss_z_back_map + self.loss_reuse_full(p_gather, z1_gather_backable)

        infonce_loss_z_back_map = link.get_world_size() * infonce_loss_z_back_map / (2 * self.lsp_level ** 2)
        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_ilsp_zbackEX_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        infonce_loss_z_back_map = (self.loss_reuse(pp1, z2_gather) + self.loss_reuse(pp2, z1_gather)) * 0.5
        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_z_back_map)


class InfoNCE_Nilsp_zbackEX_symmetric_sep(InfoNCE_halvlsp_post_sep):
    def __init__(self, i_base, temperature, lsp_level):
        super(InfoNCE_Nilsp_zbackEX_symmetric_sep, self).__init__(temperature, lsp_level, 1 - 1 / i_base)
        self.i_base = i_base
        self.i_portion = i_base - 1

    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather, z1_p_gather, z1_o_gather = concat_all_gather_isplit(z1.detach(), self.i_base, self.i_portion)
        z2_gather, z2_p_gather, z2_o_gather = concat_all_gather_isplit(z2.detach(), self.i_base, self.i_portion)

        infonce_loss_map = 0
        cross_m_to_i = torch.tensor(0)
        for p in p1:
            infonce_loss_map = infonce_loss_map + (self.loss_reuse(p[:p.size(0)//2], torch.cat([z2_p_gather, z2_o_gather], dim=0))
                                                   + self.loss_reuse(p[p.size(0)//2:], torch.cat([z1_p_gather, z1_o_gather], dim=0))) / 2
        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2

        infonce_loss_indi = (self.loss_reuse(p2[:p2.size(0)//2], torch.cat([z2_o_gather, z2_p_gather], dim=0)) + self.loss_reuse(p2[p2.size(0)//2:], torch.cat([z1_o_gather, z1_p_gather], dim=0))) / 2
        cross_i_to_m = torch.tensor(0)

        infonce_loss_z_back_map = (self.loss_reuse(pp1, z2_gather) + self.loss_reuse(pp2, z1_gather)) * 0.5
        loss = infonce_loss_map * self.sep_weight + infonce_loss_indi * (1 - self.sep_weight) + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_z_back_map)


class InfoNCE_zbackEX(InfoNCE):
    def forward_vis(self, p1, z1, pp1, p2, z2, pp2):
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = self.loss_reuse(p1, z2_gather)
        infonce_loss_indi = self.loss_reuse(p2, z1_gather)

        cross_m_to_i = self.loss_reuse(p1, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        infonce_loss_z_back_map = (self.loss_reuse(pp1, z2_gather) + self.loss_reuse(pp2, z1_gather)) * 0.5
        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_z_back_map)


class InfoNCE_ilsp_fullprojs_constrast_sep(InfoNCE_halvlsp_post_sep):
    def __init__(self, Tema, temperature, lsp_level, sep_weight=0.5, Starget='selfe'):
        super(InfoNCE_ilsp_fullprojs_constrast_sep, self).__init__(temperature, lsp_level, sep_weight)
        self.Tema = Tema
        self.Starget = Starget

    def forward_vis(self, p1, z1, ep1, ez1, eze1, p2, z2, ep2, ez2, eze2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)

        if self.Tema:
            zz1 = eze1 / eze1.norm(dim=-1, keepdim=True)
            zz2 = eze2 / eze2.norm(dim=-1, keepdim=True)
        else:
            zz1 = ez1 / ez1.norm(dim=-1, keepdim=True)
            zz2 = ez2 / ez2.norm(dim=-1, keepdim=True)
        zz1_gather = concat_all_gather(zz1.detach())
        zz2_gather = concat_all_gather(zz2.detach())

        if self.Starget == 'selfe':
            z1 = z1 / z1.norm(dim=-1, keepdim=True)
            z2 = z2 / z2.norm(dim=-1, keepdim=True)

            z1_gather = concat_all_gather(z1.detach())
            z2_gather = concat_all_gather(z2.detach())
        elif self.Starget == 'emaz':
            if self.Tema:
                z1 = ez1 / ez1.norm(dim=-1, keepdim=True)
                z2 = ez2 / ez2.norm(dim=-1, keepdim=True)

                z1_gather = concat_all_gather(z1.detach())
                z2_gather = concat_all_gather(z2.detach())
            else:
                z1_gather = zz1_gather
                z2_gather = zz2_gather
        elif self.Starget == 'emazema':
            if self.Tema:
                z1_gather = zz1_gather
                z2_gather = zz2_gather
            else:
                z1 = eze1 / eze1.norm(dim=-1, keepdim=True)
                z2 = eze2 / eze2.norm(dim=-1, keepdim=True)

                z1_gather = concat_all_gather(z1.detach())
                z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        infonce_loss_z_back_map = (self.loss_reuse(ep1, zz2_gather) + self.loss_reuse(ep2, zz1_gather)) * 0.5
        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_z_back_map)


class InfoNCE_ilsp_zbackEX_emaProj_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, pp1, zz1, p2, z2, pp2, zz2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        zz1 = zz1 / zz1.norm(dim=-1, keepdim=True)
        zz2 = zz2 / zz2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())
        zz1_gather = concat_all_gather(zz1.detach())
        zz2_gather = concat_all_gather(zz2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        infonce_loss_z_back_map = (self.loss_reuse(pp1, zz2_gather) + self.loss_reuse(pp2, zz1_gather)) * 0.5
        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_z_back_map)


class ConsSim_ilsp_zbackEX_emaProj_sep(InfoNCE_halvlsp_post_sep):
    def cosine_similarity_loss(self, p, z):
        return - self.cosine_similarity(p, z)

    def forward_vis(self, p1, z1, pp1, zz1, p2, z2, pp2, zz2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        zz1 = zz1 / zz1.norm(dim=-1, keepdim=True)
        zz2 = zz2 / zz2.norm(dim=-1, keepdim=True)

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.cosine_similarity_loss(p, z2.detach())
            cross_m_to_i = cross_m_to_i + self.cosine_similarity_loss(p, z1.detach())

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.cosine_similarity_loss(p, z1.detach())
            cross_i_to_m = cross_i_to_m + self.cosine_similarity_loss(p, z2.detach())

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        infonce_loss_z_back_map = (self.cosine_similarity_loss(pp1, zz2.detach()) + self.cosine_similarity_loss(pp2, zz1.detach())) * 0.5
        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_z_back_map)


class InfoNCE_ilsp_zbackEX_TemaProj_sep(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, pp1, zz1, p2, z2, pp2, zz2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        p2 = p2.split(p2.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = 0
        cross_i_to_m = 0
        for p in p2:
            infonce_loss_indi = infonce_loss_indi + self.loss_reuse(p, z1_gather)
            cross_i_to_m = cross_i_to_m + self.loss_reuse(p, z2_gather)

        infonce_loss_indi = infonce_loss_indi / self.lsp_level ** 2
        cross_i_to_m = cross_i_to_m / self.lsp_level ** 2

        infonce_loss_z_back_map = (self.loss_reuse(pp1, z2_gather) + self.loss_reuse(pp2, z1_gather)) * 0.5
        loss = (infonce_loss_map + infonce_loss_indi) * 0.5 + infonce_loss_z_back_map

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, infonce_loss_z_back_map)


class InfoNCE_hlvi_DropVoid(InfoNCE_halvlsp_post_sep):
    def loss_reuse_nonreduce(self, p, z_gather):
        # [N, E]

        p = p / p.norm(dim=-1, keepdim=True)

        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels, reduction='none')  #[N_local]

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = []
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map.append(self.loss_reuse_nonreduce(p, z2_gather))
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)
        infonce_loss_map = torch.stack(infonce_loss_map, dim=0).topk(self.lsp_level ** 2 - 1, dim=0, largest=False)[0].mean()

        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_WeightVoid(InfoNCE_hlvi_DropVoid):
    def __init__(self, weight, temperature, lsp_level):
        super(InfoNCE_hlvi_WeightVoid, self).__init__(temperature, lsp_level)
        self.weight = weight

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = []
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map.append(self.loss_reuse_nonreduce(p, z2_gather))
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)
        loss_map = torch.stack(infonce_loss_map, dim=0)
        infonce_loss_map = loss_map.topk(self.lsp_level ** 2 // 2, dim=0, largest=False)[0].mean() * (1 - self.weight) + loss_map.topk(self.lsp_level ** 2 // 2, dim=0, largest=True)[0].mean() * self.weight

        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_WeightVoid_p2mo(InfoNCE_hlvi_WeightVoid):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = []
        cross_m_to_i = []
        for p in p1:
            infonce_loss_map.append(self.loss_reuse_nonreduce(p, z2_gather))
            cross_m_to_i.append(self.loss_reuse_nonreduce(p, z1_gather))
        loss_map = torch.stack(infonce_loss_map, dim=0)
        cross_map = torch.stack(cross_m_to_i, dim=0)
        infonce_loss_map = loss_map.topk(self.lsp_level ** 2 // 2, dim=0, largest=False)[0].mean() * (1 - self.weight) + \
                           loss_map.topk(self.lsp_level ** 2 // 2, dim=0, largest=True)[0].mean() * self.weight

        cross_m_to_i = cross_map.topk(self.lsp_level ** 2 // 2, dim=0, largest=False)[0].mean()* (1 - self.weight) + \
                       cross_map.topk(self.lsp_level ** 2 // 2, dim=0, largest=True)[0].mean() * self.weight

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + 0.5 * (infonce_loss_indi + cross_m_to_i))

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_DropVoid_p2mo(InfoNCE_hlvi_DropVoid):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = []
        cross_m_to_i = []
        for p in p1:
            infonce_loss_map.append(self.loss_reuse_nonreduce(p, z2_gather))
            cross_m_to_i.append(self.loss_reuse_nonreduce(p, z1_gather))
        infonce_loss_map = torch.stack(infonce_loss_map, dim=0).topk(self.lsp_level ** 2 - 1, dim=0, largest=False)[0].mean()
        cross_m_to_i = torch.stack(cross_m_to_i, dim=0).topk(self.lsp_level ** 2 - 1, dim=0, largest=False)[0].mean()

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + 0.5 * (infonce_loss_indi + cross_m_to_i))

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_DropEasy(InfoNCE_hlvi_DropVoid):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = []
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map.append(self.loss_reuse_nonreduce(p, z2_gather))
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)
        infonce_loss_map = torch.stack(infonce_loss_map, dim=0).topk(self.lsp_level ** 2 - 1, dim=0)[0].mean()

        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_DropEasy_p2mo(InfoNCE_hlvi_DropVoid_p2mo):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = []
        cross_m_to_i = []
        for p in p1:
            infonce_loss_map.append(self.loss_reuse_nonreduce(p, z2_gather))
            cross_m_to_i.append(self.loss_reuse_nonreduce(p, z1_gather))
        infonce_loss_map = torch.stack(infonce_loss_map, dim=0).topk(self.lsp_level ** 2 - 1, dim=0)[0].mean()
        cross_m_to_i = torch.stack(cross_m_to_i, dim=0).topk(self.lsp_level ** 2 - 1, dim=0)[0].mean()

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + 0.5 * (infonce_loss_indi + cross_m_to_i))

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_sep_p2mo(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + 0.5 * (infonce_loss_indi + cross_m_to_i))

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_sep_p2mo_fix(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_indi + 0.5 * (infonce_loss_map + cross_m_to_i))

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_hlvi_sep_p2moO(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_indi + cross_m_to_i)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_halvlsp_sep_iop(InfoNCE_halvlsp_post_sep):
    def loss_IOP(self, logits_pos, logits_neg):
        return 0.5 * (F.cross_entropy(logits_pos, self.pos_label) + F.cross_entropy(logits_neg, self.neg_label))

    def forward_vis(self, p1, z1, p2, z2, logits):
        self.pos_label = torch.ones(p1.size(0)//2, dtype=torch.long).cuda()
        self.neg_label = torch.zeros(p1.size(0)//2, dtype=torch.long).cuda()

        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)
        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_reuse(p, z2_gather)
            cross_m_to_i = cross_m_to_i + self.loss_reuse(p, z1_gather)

        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss_reuse(p2, z1_gather)
        cross_i_to_m = self.loss_reuse(p2, z2_gather)

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        ver_loss = self.loss_IOP(logits[0], logits[1])
        hor_loss = self.loss_IOP(logits[2], logits[3])

        iop_loss = 0.5 * (ver_loss + hor_loss)
        loss += iop_loss

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m, ver_loss, hor_loss)


class InfoNCE_focal_sep(InfoNCE_halvlsp_post_sep):
    def __init__(self, temperature, lsp_level, gamma):
        super(InfoNCE_focal_sep, self).__init__(temperature, lsp_level)
        self.gamma = gamma

    def loss_focal(self, p, z):
        # [N, E]
        offset = link.get_rank() * z.size(0)

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + z.size(0)].diag()  # [N_local]
        p_z_probs_actual = p_z_probs.exp()

        loss = - torch.sum((1 - p_z_probs_actual)**self.gamma * p_z_probs) / p.size(0)
        return loss

    def forward_vis(self, p1, z1, p2, z2):
        p1 = p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)

        infonce_loss_map = 0
        cross_m_to_i = 0
        for p in p1:
            infonce_loss_map = infonce_loss_map + self.loss_focal(p, z2.detach())
            cross_m_to_i = cross_m_to_i + self.loss(p, z1.detach())
        infonce_loss_map = infonce_loss_map / self.lsp_level ** 2
        cross_m_to_i = cross_m_to_i / self.lsp_level ** 2

        infonce_loss_indi = self.loss(p2, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_focalR_sep(InfoNCE_focal_sep):
    def loss_focal(self, p, z):
        # [N, E]
        offset = link.get_rank() * z.size(0)

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + z.size(0)].diag()  # [N_local]
        p_z_probs_actual = p_z_probs.exp()

        loss = - torch.sum((1 - (0.014226437796801052 - p_z_probs_actual))**self.gamma * p_z_probs) / p.size(0)
        return loss


class InfoNCE_halvlsp_postpre(InfoNCE_halvlsp_post_sep):
    def forward_vis(self, p1, z1, p2, z2):
        p1 = sum(p1.split(p1.size(0) // self.lsp_level ** 2, dim=0)) / self.lsp_level ** 2

        infonce_loss_map = self.loss(p1, z2.detach())
        infonce_loss_indi = self.loss(p2, z1.detach())

        cross_m_to_i = self.loss(p1, z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach())

        loss = 0.5 * (infonce_loss_map + infonce_loss_indi)

        return loss, (loss, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_QUAD(InfoNCE):
    def __init__(self, temperature, weight_ori):
        super(InfoNCE_QUAD, self).__init__(temperature)
        self.weight = weight_ori

    def forward(self, p1, z1, p2, z2):
        return 0.5 * (self.weight * (self.loss(p1, z2.detach()) + self.loss(p2, z1.detach())) +
                       (1 - self.weight) * (self.loss(p2, z2.detach()) + self.loss(p1, z1.detach())))


class InfoNCE_QUAD_Queue(InfoNCE_QUAD):
    def __init__(self, K, temperature, weight_ori):
        super(InfoNCE_QUAD_Queue, self).__init__(temperature, weight_ori)
        self.K = K

    def loss(self, p, z_full):
        offset = link.get_rank() * p.shape[0]
        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        p_z_m = p.mm(z_full) / self.temperature  #[N_local, N+K]

        return F.cross_entropy(p_z_m, labels)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z1_gather, queue_1, z2_gather, queue_2, queue_ptr):
        batch_size = z1_gather.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue_1[:, ptr:ptr + batch_size] = z1_gather.T
        queue_2[:, ptr:ptr + batch_size] = z2_gather.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr

    def forward(self, *kwargs):
        raise

    def forward_vis_quad(self, p1, z1, p2, z2, queue_1, queue_2, queue_ptr):
        p1 = p1 / p1.norm(dim=-1, keepdim=True)
        p2 = p2 / p2.norm(dim=-1, keepdim=True)
        z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = z2 / z2.norm(dim=-1, keepdim=True)

        z1_gather = concat_all_gather(z1.detach())
        z2_gather = concat_all_gather(z2.detach())

        z1_full = torch.cat([z1_gather.T, queue_1.clone().detach()], dim=1)
        z2_full = torch.cat([z2_gather.T, queue_2.clone().detach()], dim=1)

        self._dequeue_and_enqueue(z1_gather, queue_1, z2_gather, queue_2, queue_ptr)

        infonce_loss_map = self.loss(p1, z2_full)
        infonce_loss_indi = self.loss(p2, z1_full)

        cross_m_to_i = self.loss(p1, z1_full)
        cross_i_to_m = self.loss(p2, z2_full)

        loss = 0.5 * (self.weight * (infonce_loss_map + infonce_loss_indi) +
                      (1 - self.weight) * (cross_i_to_m + cross_m_to_i))

        return loss, ((infonce_loss_map + infonce_loss_indi) * 0.5, infonce_loss_map, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class InfoNCE_cifar_test(InfoNCE):
    def loss(self, p, z):
        # [N, E]
        N = p.shape[0]
        p = p[:64]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)
        offset = link.get_rank() * N
        labels = torch.arange(offset, offset + 64, dtype=torch.long).cuda()
        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        return F.cross_entropy(p_z_m, labels)


class HLV_CCL(InfoNCE):
    def __init__(self, temperature, h=8, w=8, k=4, s=2, label_square=1):
        super(HLV_CCL, self).__init__(temperature)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        width = int(w*k/s-(k/s-1))
        stride = int(k/s)
        self.indexer = torch.cat([torch.arange(0, width, stride) + width * stride * i for i in range(h)], dim=0)
        self.label_square = label_square

    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels = labels**self.label_square
        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.loss_ccl(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach()))

    def forward_vis_quad(self, p1, z1, p2, z2, labels):
        infonce_loss_ccl = self.loss_ccl(p1, z2.detach(), labels.clone())
        infonce_loss_indi = self.loss(p2, z1.detach())

        cross_m_to_i = self.loss(p1[self.indexer], z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach()[self.indexer])

        loss = 0.5 * (self.weight * (infonce_loss_ccl + infonce_loss_indi) +
                       (1 - self.weight) * (cross_i_to_m + cross_m_to_i))

        return loss, ((infonce_loss_ccl + infonce_loss_indi) * 0.5, infonce_loss_ccl, infonce_loss_indi, cross_m_to_i, cross_i_to_m)

    def forward_vis_five(self, p1, z1, p2, z2, labels):
        infonce_loss_ccl = self.loss_ccl(p1, z2.detach(), labels.clone())
        infonce_loss_bindi = self.loss(p1[self.indexer], z2.detach()[self.indexer])
        infonce_loss_indi = self.loss(p2, z1.detach())

        cross_m_to_i = self.loss(p1[self.indexer], z1.detach())
        cross_i_to_m = self.loss(p2, z2.detach()[self.indexer])

        return 0.5 * (self.weight * ((self.weight_ccl * infonce_loss_ccl +
                                       (1 - self.weight_ccl) * infonce_loss_bindi) +
                                      infonce_loss_indi) +
                       (1 - self.weight) * (cross_i_to_m + cross_m_to_i)), \
               (((self.weight_ccl * infonce_loss_ccl + (1 - self.weight_ccl) * infonce_loss_bindi) + infonce_loss_indi) * 0.5,
                infonce_loss_ccl, infonce_loss_bindi, infonce_loss_indi, cross_m_to_i, cross_i_to_m)

    def forward_vis_six(self, p1, z1, p2, z2, labels):
        infonce_loss_ccl = self.loss_ccl(p1, z2.detach(), labels.clone())
        infonce_loss_bindi = self.loss(p1[self.indexer], z2.detach()[self.indexer])

        infonce_loss_iccl = self.loss_ccl(p2, z1.detach(), labels.clone())
        infonce_loss_indi = self.loss(p2[self.indexer], z1.detach()[self.indexer])

        cross_m_to_i = self.loss(p1[self.indexer], z1.detach()[self.indexer])
        cross_i_to_m = self.loss(p2[self.indexer], z2.detach()[self.indexer])

        return 0.5 * (self.weight * ((self.weight_ccl * infonce_loss_ccl +
                                       (1 - self.weight_ccl) * infonce_loss_bindi) +
                                     (self.weight_ccl * infonce_loss_iccl +
                                      (1 - self.weight_ccl) * infonce_loss_indi)
                                     ) +
                       (1 - self.weight) * (cross_i_to_m + cross_m_to_i)), \
               (((self.weight_ccl * infonce_loss_ccl + (1 - self.weight_ccl) * infonce_loss_bindi) + (self.weight_ccl * infonce_loss_iccl + (1 - self.weight_ccl) * infonce_loss_indi)) * 0.5,
                infonce_loss_ccl, infonce_loss_bindi, infonce_loss_iccl, infonce_loss_indi, cross_m_to_i, cross_i_to_m)


class HLV_CCL_QUAD(HLV_CCL):
    def __init__(self, temperature, weight_ori, h=8, w=8, k=4, s=2, label_square=1):
        super(HLV_CCL_QUAD, self).__init__(temperature, h, w, k, s, label_square)
        self.weight = weight_ori

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * (self.loss_ccl(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach())) +
                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) + self.loss(p1[self.indexer], z1.detach())))


class HLV_CCL_NEGAMB(HLV_CCL):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m, labels)


class HLV_CCL_NEGAMB_QUAD(HLV_CCL_NEGAMB):
    def __init__(self, temperature, weight_ori, h=8, w=8, k=4, s=2, label_square=1):
        super(HLV_CCL_NEGAMB_QUAD, self).__init__(temperature, h, w, k, s, label_square)
        self.weight = weight_ori

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * (self.loss_ccl(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach())) +
                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) + self.loss(p1[self.indexer], z1.detach())))


class HLV_CCL_NEGAMB_FIVE(HLV_CCL_NEGAMB):
    def __init__(self, temperature, weight_ori, weight_ccl, h=8, w=8, k=4, s=2):
        super(HLV_CCL_NEGAMB_FIVE, self).__init__(temperature, h, w, k, s)
        self.weight = weight_ori
        self.weight_ccl = weight_ccl

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * ((self.weight_ccl * self.loss_ccl(p1, z2.detach(), labels.clone()) + (1 - self.weight_ccl) * self.loss(p1[self.indexer], z2.detach()[self.indexer])) +
                                      self.loss(p2, z1.detach())) +

                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) +
                                            self.loss(p1[self.indexer], z1.detach())))


class HLV_CCL_NEGAMB_FIVE_TL1(HLV_CCL_NEGAMB_FIVE):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m[self.indexer], labels[self.indexer])


class HLV_CCL_NEGAMB_FIVE_TL1_EXT(HLV_CCL_NEGAMB_FIVE_TL1):
    def __init__(self, temperature, temperature_ex, weight_ori, weight_ccl, h=8, w=8, k=4, s=2):
        super(HLV_CCL_NEGAMB_FIVE_TL1_EXT, self).__init__(temperature, weight_ori, weight_ccl, h, w, k, s)
        self.temperature_ex = temperature_ex

    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature_ex  #[N_local, N]

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m[self.indexer], labels[self.indexer])


class HLV_CCL_NEGAMB_QUAD_SAMPLE(HLV_CCL_NEGAMB_QUAD):
    def __init__(self, neg_size, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_NEGAMB_QUAD_SAMPLE, self).__init__(temperature, weight_ori, h, w, k, s)
        self.neg_size = neg_size

    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        l_0 = labels < 1
        l_0_back = l_0.clone()
        for i, l_i in enumerate(l_0_back):
            l_i_0_idx = l_i.nonzero()
            l_0[i][l_i_0_idx[torch.randperm(len(l_i_0_idx))][:self.neg_size - 1]] = False

        for rank in range(link.get_world_size()):
            if rank != link.get_rank():
                _offset = rank * p.size(0)
                _l_0 = torch.ones_like(l_0).bool()
                for i in range(_l_0.size(0)):
                    _l_0[i][torch.randperm(_l_0.size(1))[:self.neg_size]] = False
                p_z_m[:, _offset:_offset + p.size(0)].masked_fill_(_l_0,  -10)

        p_z_m[:, offset:offset + p.size(0)].masked_fill_(l_0, -10)

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m, labels)


class HLV_CCL_NEGAMB_QUAD_SAMPLE_B(HLV_CCL_NEGAMB_QUAD_SAMPLE):
    def __init__(self, c_pos_size, neg_size, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_NEGAMB_QUAD_SAMPLE_B, self).__init__(neg_size, temperature, weight_ori, h, w, k, s)
        self.c_pos_size = c_pos_size

    def loss_ccl(self, p, z, labels):
        p_size_0 = p.size(0)
        offset = link.get_rank() * p_size_0

        indexer_sample = torch.stack([x for x in torch.arange(p.size(0)) if x not in self.indexer])
        indexer_sample = indexer_sample[torch.randperm(len(indexer_sample))][:self.c_pos_size]
        indexer_sample = torch.cat([self.indexer, indexer_sample], dim=0)
        # [N, E]
        p = p[indexer_sample]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        labels = labels[indexer_sample]
        l_0 = labels < 1
        l_0_back = l_0.clone()
        for i, l_i in enumerate(l_0_back):
            l_i_0_idx = l_i.nonzero()
            l_0[i][l_i_0_idx[torch.randperm(len(l_i_0_idx))][:self.neg_size - 1]] = False

        for rank in range(link.get_world_size()):
            if rank != link.get_rank():
                _offset = rank * p_size_0
                _l_0 = torch.ones_like(l_0).bool()
                for i in range(_l_0.size(0)):
                    _l_0[i][torch.randperm(_l_0.size(1))[:self.neg_size]] = False
                p_z_m[:, _offset:_offset + p_size_0].masked_fill_(_l_0,  -10)

        p_z_m[:, offset:offset + p_size_0].masked_fill_(l_0, -10)

        labels = torch.arange(offset, offset + p_size_0, dtype=torch.long).cuda()[indexer_sample]

        return F.cross_entropy(p_z_m, labels)


class HLV_OCCL_NEGAMB_QUAD(HLV_CCL_NEGAMB_QUAD):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        loss = F.cross_entropy(p_z_m, labels)

        return loss if loss > -20 else loss / loss.detach() * -7


class HLV_CCL_NEGAMB_QUAD_EXT(HLV_CCL_NEGAMB_QUAD):
    def __init__(self, temperature, temperature_ex, weight_ori, h=8, w=8, k=4, s=2, label_square=1):
        super(HLV_CCL_NEGAMB_QUAD_EXT, self).__init__(temperature, weight_ori, h, w, k, s, label_square)
        self.temperature_ex = temperature_ex

    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature_ex  #[N_local, N]

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m, labels)


class HLV_CCL_POSAMB(HLV_CCL):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels.masked_fill_(labels > 0, 1.)
        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss


class HLV_CCL_POSAMB_QUAD(HLV_CCL_POSAMB):
    def __init__(self, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_POSAMB_QUAD, self).__init__(temperature, h, w, k, s)
        self.weight = weight_ori

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * (self.loss_ccl(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach())) +
                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) + self.loss(p1[self.indexer], z1.detach())))


class HLV_CCL_SIGPOS(HLV_CCL):
    def loss_ccl(self, p, z, labels):
        ones = torch.ones_like(labels)
        ones[:, self.indexer] = 0
        ones = ones.bool()

        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_m[:, offset:offset + p.size(0)].masked_fill_(torch.logical_and(labels>0, ones), -10)
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels.masked_fill_(torch.logical_and(labels>0, ones), 0.)
        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss


class HLV_CCL_SIGPOS_QUAD(HLV_CCL_SIGPOS):
    def __init__(self, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_SIGPOS_QUAD, self).__init__(temperature, h, w, k, s)
        self.weight = weight_ori

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * (self.loss_ccl(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach())) +
                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) + self.loss(p1[self.indexer], z1.detach())))


class HLV_CCL_MASKAMB(HLV_CCL):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_m[:, offset:offset + p.size(0)].masked_fill_(torch.logical_and(labels>0, labels<1), -10)

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m, labels)


class HLV_CCL_MASKAMB_FIVE(HLV_CCL_MASKAMB):
    def __init__(self, temperature, weight_ori, weight_ccl, h=8, w=8, k=4, s=2):
        super(HLV_CCL_MASKAMB_FIVE, self).__init__(temperature, h, w, k, s)
        self.weight = weight_ori
        self.weight_ccl = weight_ccl

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * ((self.weight_ccl * self.loss_ccl(p1, z2.detach(), labels.clone()) + (1 - self.weight_ccl) * self.loss(p1[self.indexer], z2.detach()[self.indexer])) +
                                      self.loss(p2, z1.detach())) +

                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) +
                                            self.loss(p1[self.indexer], z1.detach())))


class HLV_CCL_MASKAMB_FIVE_TL1(HLV_CCL_MASKAMB_FIVE):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_m[:, offset:offset + p.size(0)].masked_fill_(torch.logical_and(labels>0, labels<1), -10)

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m[self.indexer], labels[self.indexer])


class HLV_CCL_MASKAMB_SIX_TL1(HLV_CCL_MASKAMB_FIVE):
    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * ((self.weight_ccl * self.loss_ccl(p1, z2.detach(), labels.clone()) + (1 - self.weight_ccl) * self.loss(p1[self.indexer], z2.detach()[self.indexer])) +
                                     (self.weight_ccl * self.loss_ccl(p2, z1.detach(), labels.clone()) + (1 - self.weight_ccl) * self.loss(p2[self.indexer], z1.detach()[self.indexer]))) +

                       (1 - self.weight) * (self.loss(p2[self.indexer], z2.detach()[self.indexer]) +
                                            self.loss(p1[self.indexer], z1.detach()[self.indexer])))
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_m[:, offset:offset + p.size(0)].masked_fill_(torch.logical_and(labels>0, labels<1), -10)

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m[self.indexer], labels[self.indexer])


class HLV_CCL_MASKAMB_QUAD(HLV_CCL_MASKAMB):
    def __init__(self, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_MASKAMB_QUAD, self).__init__(temperature, h, w, k, s)
        self.weight = weight_ori

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * (self.loss_ccl(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach())) +
                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) + self.loss(p1[self.indexer], z1.detach())))


class HLV_CCL_MASKAMB_QUAD_SAMPLE(HLV_CCL_MASKAMB_QUAD):
    def __init__(self, neg_size, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_MASKAMB_QUAD_SAMPLE, self).__init__(temperature, weight_ori, h, w, k, s)
        self.neg_size = neg_size

    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        l_0 = labels == 0
        l_0_back = l_0.clone()
        for i, l_i in enumerate(l_0_back):
            l_i_0_idx = l_i.nonzero()
            l_0[i][l_i_0_idx[torch.randperm(len(l_i_0_idx))][:self.neg_size - 1]] = False

        for rank in range(link.get_world_size()):
            if rank != link.get_rank():
                _offset = rank * p.size(0)
                _l_0 = torch.ones_like(l_0).bool()
                for i in range(_l_0.size(0)):
                    _l_0[i][torch.randperm(_l_0.size(1))[:self.neg_size]] = False
                p_z_m[:, _offset:_offset + p.size(0)].masked_fill_(_l_0,  -10)

        p_z_m[:, offset:offset + p.size(0)].masked_fill_(torch.logical_or(torch.logical_and(labels>0, labels<1), l_0), -10)

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m, labels)


class HLV_CCL_MASKAMB_QUAD_SAMPLE_B(HLV_CCL_MASKAMB_QUAD_SAMPLE):
    def __init__(self, c_pos_size, neg_size, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_MASKAMB_QUAD_SAMPLE_B, self).__init__(neg_size, temperature, weight_ori, h, w, k, s)
        self.c_pos_size = c_pos_size

    def loss_ccl(self, p, z, labels):
        p_size_0 = p.size(0)
        offset = link.get_rank() * p_size_0

        indexer_sample = torch.stack([x for x in torch.arange(p.size(0)) if x not in self.indexer])
        indexer_sample = indexer_sample[torch.randperm(len(indexer_sample))][:self.c_pos_size]
        indexer_sample = torch.cat([self.indexer, indexer_sample], dim=0)
        # [N, E]
        p = p[indexer_sample]

        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]

        labels = labels[indexer_sample]
        l_0 = labels == 0
        l_0_back = l_0.clone()
        for i, l_i in enumerate(l_0_back):
            l_i_0_idx = l_i.nonzero()
            l_0[i][l_i_0_idx[torch.randperm(len(l_i_0_idx))][:self.neg_size - 1]] = False

        for rank in range(link.get_world_size()):
            if rank != link.get_rank():
                _offset = rank * p_size_0
                _l_0 = torch.ones_like(l_0).bool()
                for i in range(_l_0.size(0)):
                    _l_0[i][torch.randperm(_l_0.size(1))[:self.neg_size]] = False
                p_z_m[:, _offset:_offset + p_size_0].masked_fill_(_l_0,  -10)

        p_z_m[:, offset:offset + p_size_0].masked_fill_(torch.logical_or(torch.logical_and(labels>0, labels<1), l_0), -10)

        labels = torch.arange(offset, offset + p_size_0, dtype=torch.long).cuda()[indexer_sample]

        return F.cross_entropy(p_z_m, labels)


class HLV_CCL_MASKAMB_QUAD_TL1(HLV_CCL_MASKAMB_QUAD):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_m[:, offset:offset + p.size(0)].masked_fill_(torch.logical_and(labels>0, labels<1), -10)

        labels = torch.arange(offset, offset + p.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(p_z_m[self.indexer], labels[self.indexer])


class HLV_CCL_TL1(HLV_CCL):
    def loss_ccl(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)[self.indexer]) / 64
        return loss


class HLV_CCL_TL2(HLV_CCL):
    def loss_ccl(self, p, z, labels):
        z = z[self.indexer]
        labels = labels[:, self.indexer]

        offset = link.get_rank() * z.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + z.size(0)]  #[N_local, N_local]

        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss


class HLV_CCL_FIVE_TL2(HLV_CCL_TL2):
    def __init__(self, temperature, weight_ori, weight_ccl, h=8, w=8, k=4, s=2):
        super(HLV_CCL_FIVE_TL2, self).__init__(temperature, h, w, k, s)
        self.weight = weight_ori
        self.weight_ccl = weight_ccl


class HLV_CCL_TL2_QUAD(HLV_CCL_TL2):
    def __init__(self, temperature, weight_ori, h=8, w=8, k=4, s=2):
        super(HLV_CCL_TL2_QUAD, self).__init__(temperature, h, w, k, s)
        self.weight = weight_ori

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.weight * (self.loss_ccl(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach())) +
                       (1 - self.weight) * (self.loss(p2, z2.detach()[self.indexer]) + self.loss(p1[self.indexer], z1.detach())))


class CCL(InfoNCE):
    def __init__(self, temperature):
        super(CCL, self).__init__(temperature)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def loss(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss

    def forward(self, p1, z1, p2, z2, labels):
        return 0.5 * (self.loss(p1, z2.detach(), labels.clone()) + self.loss(p2, z1.detach(), labels.clone()))


class CCL_test(CCL):
    def __init__(self, temperature):
        super(CCL_test, self).__init__(temperature)
        self.indexer = torch.cat([torch.arange(0, 15, 2) + 30 * i for i in range(8)], dim=0)

    def loss(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)[self.indexer]) / 64
        return loss


class CCL_test2(CCL_test):
    def loss(self, p, z, labels):
        offset = link.get_rank() * p.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels = labels[:, self.indexer]
        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs[:, self.indexer].mul(labels)) / p.size(0)
        return loss


class CCL_test2_fix(CCL_test):
    def loss(self, p, z, labels):
        z = z[self.indexer]
        labels = labels[:, self.indexer]

        offset = link.get_rank() * z.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + z.size(0)]  #[N_local, N_local]

        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss


class CCL_test_sample(CCL_test):
    def __init__(self, temperature, con_size):
        super(CCL_test_sample, self).__init__(temperature)
        self.con_size = con_size

        self.indexer_sample = torch.stack([x for x in torch.arange(225) if x not in self.indexer])

    def loss(self, p, z, labels):
        indexer_sample = self.indexer_sample[torch.randperm(len(self.indexer_sample))][:self.con_size]
        indexer_sample = torch.cat([self.indexer, indexer_sample], dim=0)
        p = p[indexer_sample]
        z = z[self.indexer]
        labels = labels[:, self.indexer][indexer_sample]

        offset = link.get_rank() * z.size(0)

        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local_sample, N_target_all]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + z.size(0)]  #[N_local_sample, N_target]

        labels /= labels.sum(dim=1, keepdim=True)

        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss


class CCL_test_sample_test(CCL_test):
    def loss(self, p, z, labels):
        p = p[self.indexer]
        z = z[self.indexer]
        labels = labels[:, self.indexer][self.indexer]

        assert p.size(0) == 64

        offset = link.get_rank() * p.size(0)
        # [N, E]
        p = p / p.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)

        z_gather = concat_all_gather(z)

        p_z_m = p.mm(z_gather.T) / self.temperature  #[N_local, N]
        p_z_probs = self.logsoft(p_z_m)[:, offset:offset + p.size(0)]  #[N_local, N_local]

        labels /= labels.sum(dim=1, keepdim=True)
        loss = - torch.sum(p_z_probs.mul(labels)) / p.size(0)
        return loss


@torch.no_grad()
def concat_all_gather(tensor):
    """gather the given tensor"""

    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    dist.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def concat_all_gather_woself(tensor):
    """gather the given tensor"""

    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    dist.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather[:link.get_rank()] + tensors_gather[link.get_rank() + 1:], dim=0)
    return output


@torch.no_grad()
def concat_all_gather_isplit(tensor, i_base, i_portion):
    """gather the given tensor"""

    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    dist.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    output_p = torch.cat([z[:(z.size(0)//i_base)*i_portion] for z in tensors_gather], dim=0)
    output_o = torch.cat([z[(z.size(0)//i_base)*i_portion:] for z in tensors_gather], dim=0)
    return output, output_p, output_o


@torch.no_grad()
def concat_all_gather_4sp_isplit(tensor, i_base, i_portion, lsp_level):
    """gather the given tensor"""
    tensor_split_size = tensor.size(0) // lsp_level ** 2

    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    dist.all_gather(tensors_gather, tensor)

    tensors_gather = [t.split(tensor_split_size, dim=0) for t in tensors_gather]


    output_ps = [torch.cat([z[:(z.size(0)//i_base)*i_portion] for z in tensors_gather[t_th]], dim=0) for t_th in range(lsp_level ** 2)]
    output_os = [torch.cat([z[(z.size(0)//i_base)*i_portion:] for z in tensors_gather[t_th]], dim=0) for t_th in range(lsp_level ** 2)]
    return output_ps, output_os
