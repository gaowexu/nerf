#!/usr/bin/python3
#
# ------------------------------------------------------------------------------
# Author: Gaowei Xu (gaowexu1991@gmail.com)
# ------------------------------------------------------------------------------
import torch


def get_mse_loss(rgb_map: torch.Tensor, gt_rgb_map: torch.Tensor):
    """
    Mean square error loss.

    :param rgb_map: torch.Tensor with shape (B, 3), 3 indicates r, g, b.
    :param gt_rgb_map: torch.Tensor with shape (B, 3), 3 indicates r, g, b.

    :return: torch.Tensor, such as tensor(0.1658)
    """
    return torch.mean((rgb_map - gt_rgb_map) ** 2)


def get_psnr(mse: torch.Tensor):
    """
    Peak signal-to-noise ratio (PSNR) loss.

    :param mse: torch.Tensor with scalar value.
    :return: torch.Tensor with shape (1, ).
    """
    return -10.0 * torch.log(mse) / torch.log(torch.Tensor([10.0]).to(mse.device))


if __name__ == "__main__":
    B = 1024
    rgb = torch.rand(B, 3)
    rgb_gt = torch.rand(B, 3)

    mse = get_mse_loss(rgb_map=rgb, gt_rgb_map=rgb_gt)
    print(mse, type(mse))
    print("mse.shape = {}".format(mse.shape))

    psnr = get_psnr(mse=mse)
    print(psnr, type(psnr))
    print("psnr.shape = {}".format(psnr.shape))






