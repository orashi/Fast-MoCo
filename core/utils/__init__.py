from .fastmoco_builder import FastMoCo


def unsupervised_entry(config):
    if config['type'] not in globals():
        raise NotImplementedError
    return globals()[config['type']](**config['kwargs'])