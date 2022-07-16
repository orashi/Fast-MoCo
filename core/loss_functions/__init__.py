from .infoNCE import *
from .debug import InfoNCE_ilsp_sep_plus_combs_clean


def loss_entry(config):
    if type(config) is str:
        if config not in globals():
            raise NotImplementedError
        return globals()[config]()
    else:
        if config['type'] not in globals():
            raise NotImplementedError
        return globals()[config['type']](**config['kwargs'])