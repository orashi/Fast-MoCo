# Fast-MoCo: Boost Momentum-based Contrastive Learning with Combinatorial Patches

# Usage

## Preparation

Install all required dependencies in requirements.txt and replace all `..path/..to` in the code to the absolute path
to corresponding resources.

*Only slurm-based distributed training support is implemented in this repo.

## Experiments

All experiment files are located in `ssl_exp/fast_moco` folder.

To perform self-supervised training on ImageNet, run:

```bash
sh run.sh  <gpu-partition> 
```

To perform linear evaluation, make sure the pretrained weight is located in ./checkpoints folder or specified in
config_finetune.yaml (saver.pretrain.path), run:

```bash
sh run_finetune.sh  <gpu-partition> 
```

# Pretrained Models

| Backbone  | Model       | Pretrained <br/>Epochs | Top-1 Linear <br/>Evaluation Accuracy | Pretrained Weight                                                                                | md5 |
|-----------|-------------|-------------------|----------------------------------|--------------------------------------------------------------------------------------------------|-----|
| Resnet-50 | Fast-MoCo   | 100               | 73.5%                            | [checkpoint](https://drive.google.com/file/d/12ZEKiUg8ep2LgX5cJbEELmPHJRcFEMrR/view?usp=sharing) |   d6ea9023372c14db94b0dc285f216f99  |
| Resnet-50 | Fast-MoCo   | 200               | 75.1%                            | [checkpoint](https://drive.google.com/file/d/1dLRm3Ba8qgK3iKxmyZ4WaDzAJM64z7yx/view?usp=sharing) |  9f1c29ea305d9214f265fa460856db28   |
| Resnet-50 | Fast-MoCo   | 400               | 75.5%                            | [checkpoint](https://drive.google.com/file/d/1dluLLYPkpbYTZ4LxLbtvjsi-bMDBsrFm/view?usp=sharing)  |  79ae2aff26c6cb762feaf9155b137d4a   |

