from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from .datasets import ImageNetDataset
from .transforms import build_transformer, TwoCropsTransform, GaussianBlur
from .auto_augmentation import ImageNetPolicy
from .sampler import build_sampler
from .metrics import build_evaluator

def build_common_augmentation(aug_type):
    """
    common augmentation settings for training/testing ImageNet
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug_type == 'STANDARD':
        augmentation = [
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'SEMI':
        augmentation = [
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'AUTOAUG':
        augmentation = [
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'MOCOV1':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.), interpolation=Image.BICUBIC),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'MOCOV2':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.), interpolation=Image.BICUBIC),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'LINEAR':
        augmentation = [
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'ONECROP':
        augmentation = [
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        raise RuntimeError("undefined augmentation type for ImageNet!")

    if aug_type in ['MOCOV1', 'MOCOV2']:
        return TwoCropsTransform(transforms.Compose(augmentation))
    else:
        return transforms.Compose(augmentation)


def build_imagenet_train_dataloader(cfg_dataset, data_type='train'):
    """
    build training dataloader for ImageNet
    """
    cfg_train = cfg_dataset['train']
    # build dataset
    image_reader = cfg_dataset[data_type].get('image_reader', {})
    # PyTorch data preprocessing
    if isinstance(cfg_train['transforms'], list):
        transformer = build_transformer(cfgs=cfg_train['transforms'],
                                        image_reader=image_reader)
    else:
        transformer = build_common_augmentation(cfg_train['transforms']['type'])
    dataset = ImageNetDataset(
        root_dir=cfg_train['root_dir'],
        meta_file=cfg_train['meta_file'],
        transform=transformer,
        read_from=cfg_dataset['read_from'],
        image_reader_type=image_reader.get('type', 'pil'),
        server_cfg=cfg_train.get("server", {}),
    )

    # build sampler
    cfg_train['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_train['sampler'], cfg_dataset)
    if cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}
    # build dataloader
    # PyTorch dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset['batch_size'],
        shuffle=False,
        num_workers=cfg_dataset['num_workers'],
        pin_memory=True,
        sampler=sampler
    )
    return {'type': 'train', 'loader': loader}


def build_imagenet_test_dataloader(cfg_dataset, data_type='test'):
    """
    build testing/validation dataloader for ImageNet
    """
    cfg_test = cfg_dataset['test']
    # build evaluator
    evaluator = None
    if cfg_test.get('evaluator', None):
        evaluator = build_evaluator(cfg_test['evaluator'])

    image_reader = cfg_dataset[data_type].get('image_reader', {})
    # PyTorch data preprocessing
    if isinstance(cfg_test['transforms'], list):
        transformer = build_transformer(cfgs=cfg_test['transforms'],
                                        image_reader=image_reader)
    else:
        transformer = build_common_augmentation(cfg_test['transforms']['type'])
    dataset = ImageNetDataset(
        root_dir=cfg_test['root_dir'],
        meta_file=cfg_test['meta_file'],
        transform=transformer,
        read_from=cfg_dataset['read_from'],
        evaluator=evaluator,
        image_reader_type=image_reader.get('type', 'pil'),
    )
    # build sampler
    assert cfg_test['sampler'].get('type', 'distributed') == 'distributed'
    cfg_test['sampler']['kwargs'] = {'dataset': dataset, 'round_up': False}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_test['sampler'], cfg_dataset)
    # build dataloader
    # PyTorch dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset['batch_size'],
        shuffle=False,
        num_workers=cfg_dataset['num_workers'],
        pin_memory=cfg_dataset['pin_memory'],
        sampler=sampler
    )
    return {'type': 'test', 'loader': loader}


