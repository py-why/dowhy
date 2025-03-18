# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
The functions in this file are borrowed from DomainBed: https://github.com/facebookresearch/DomainBed
    @inproceedings{gulrajani2021in,
     title={In Search of Lost Domain Generalization},
     author={Ishaan Gulrajani and David Lopez-Paz},
     booktitle={International Conference on Learning Representations},
     year={2021},
    }
"""

import torch


def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res.clamp_min_(1e-30)


def gaussian_kernel(x, y, gamma):
    D = my_cdist(x, y)
    K = torch.zeros_like(D)

    K.add_(torch.exp(D.mul(-gamma)))

    return K


def mmd_compute(x, y, kernel_type, gamma):
    if kernel_type == "gaussian":
        Kxx = gaussian_kernel(x, x, gamma).mean()
        Kyy = gaussian_kernel(y, y, gamma).mean()
        Kxy = gaussian_kernel(x, y, gamma).mean()
        return Kxx + Kyy - 2 * Kxy
    else:
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff
