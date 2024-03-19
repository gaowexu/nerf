#!/usr/bin/python3
#
# ------------------------------------------------------------------------------
# Author: Gaowei Xu (gaowexu1991@gmail.com)
# ------------------------------------------------------------------------------
import torch
from nerf import NeRF
import torch.nn.functional as F
from data_loader import NeRFDataset, Phase, worker_init_fn
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from loss import get_mse_loss, get_psnr


class Predictor(object):
    def __init__(self, config: dict):
        self._config = config

        self._batch_size = 1000
        self._models_root_dir = self._config["ModelZooDir"]
        self._distance_min = self._config["MinDistance"]
        self._distance_max = self._config["MaxDistance"]
        self._n_samples_coarse = self._config["NOfCoarse"]
        self._perturb = self._config["Perturb"]
        self._n_samples_fine = self._config["NOfFine"]

        self._coarse_nerf_model_path = self._config["CoarseModelFullPath"]
        self._fine_nerf_model_path = self._config["FineModelFullPath"]

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._coarse_model = torch.load(self._coarse_nerf_model_path)
        self._coarse_model = self._coarse_model.to(self._device)

        self._fine_model = torch.load(self._fine_nerf_model_path)
        self._fine_model = self._fine_model.to(self._device)

        self._coarse_model.eval()
        self._fine_model.eval()

        self._val_data = NeRFDataset(phase=Phase.Validation)
        self._val_loader = DataLoader(
            dataset=self._val_data,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=True,
            worker_init_fn=partial(worker_init_fn, seed=None),
            num_workers=8,
            collate_fn=self._val_data.collate_batch,
        )

        self._mse_loss = get_mse_loss

    def decode(
        self,
        color_rgb: torch.Tensor,
        sigma: torch.Tensor,
        depth: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
    ):
        """
        Decode NeRF model's predictions to semantically meaningful values.

        :param color_rgb: torch.Tensor with shape (B, n_samples, 3).
        :param sigma: torch.Tensor with shape (B, n_samples, 1).
        :param depth: torch.Tensor with shape (B, n_samples).
        :param rays_d: torch.Tensor with shape (B, 3).
        :param rays_d: torch.Tensor with shape (B, 3).
        :param raw_noise_std: noise std.

        :return:
            rgb_map: torch.Tensor with shape (B, 3). Estimated RGB color of a ray.
            disp_map: torch.Tensor with shape (B, ). Disparity map. Inverse of depth map.
            acc_map: torch.Tensor with shape (B, ). Sum of weights along each ray.
            weights: torch.Tensor with shape (B, n_samples). Weights assigned to each sampled color.
            depth_map: torch.Tensor with shape (B, ). Estimated distance to object.
        """
        # dists.shape = (B, n_samples - 1)
        dists = depth[..., 1:] - depth[..., :-1]

        # dists.shape = (B, n_samples)
        dists = torch.cat(
            [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(self._device)],
            dim=-1,
        )

        # dists.shape = (B, n_samples)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # color_rgb.shape = (B, n_samples, 3)
        color_rgb = torch.sigmoid(color_rgb)

        # (B, n_samples, 1) --> (B, n_samples)
        sigma = torch.squeeze(sigma, dim=-1)

        noise = 0.0
        if raw_noise_std > 0.0:
            noise = torch.randn(sigma.shape) * raw_noise_std

        # alpha.shape = (B, n_samples)
        alpha = 1.0 - torch.exp(-F.relu(sigma + noise) * dists)

        # weights.shape = (B, n_samples)
        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [
                        torch.ones((alpha.shape[0], 1)).to(self._device),
                        1.0 - alpha + 1e-10,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )[:, :-1]
        )

        # rgb_map.shape = (B, 3)
        rgb_map = torch.sum(weights[..., None] * color_rgb, dim=-2)

        # weights.shape = (B, n_samples), depth.shape = (B, n_samples)
        # depth_map.shape = (B, )
        depth_map = torch.sum(weights * depth, dim=-1)

        # disp_map.shape = (B, )
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )

        # acc_map.shape = (B, )
        acc_map = torch.sum(weights, dim=-1)

        return rgb_map, disp_map, acc_map, weights, depth_map

    # Hierarchical sampling (section 5.2)
    def sample_pdf(
        self, bins: torch.Tensor, weights: torch.Tensor, n_samples_fine: int, det=False
    ):
        """
        Probability Density Function (PDF) generation.

        :param bins: torch.Tensor with shape (B, n_samples_coarse - 1)
        :param weights: torch.Tensor with shape (B, n_samples_coarse - 2)
        :param n_samples_fine: Sampling points along each ray for refinement phase.
        :param det: flag. True or False, defaults to False.
        :return:
        """
        weights = weights + 1e-5  # prevent NaNs

        # (B, n_samples_coarse - 2)
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

        # (B, n_samples_coarse - 2)
        cdf = torch.cumsum(pdf, dim=-1)

        # (B, n_samples_coarse - 1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Take uniform samples
        if det:
            u = torch.linspace(0.0, 1.0, steps=n_samples_fine)  # (n_samples_fine, )
            u = u.expand(list(cdf.shape[:-1]) + [n_samples_fine])  # (B, n_samples_fine)
        else:
            u = torch.rand(
                list(cdf.shape[:-1]) + [n_samples_fine]
            )  # (B, n_samples_fine)

        # Invert CDF
        u = u.contiguous().to(self._device)  # (B, n_samples_fine)
        inds = torch.searchsorted(cdf, u, right=True)  # (B, n_samples_fine)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)  # (B, n_samples_fine)
        above = torch.min(
            (cdf.shape[-1] - 1) * torch.ones_like(inds), inds
        )  # (B, n_samples_fine)
        inds_g = torch.stack([below, above], -1)  # (B, n_samples_fine, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def render_rays(
        self,
        rays: torch.Tensor,
        n_samples_coarse: int = 64,
        perturb: float = 1.0,
        n_samples_fine: int = 128,
    ):
        """
        Volumetric rendering.

        :param rays: torch.Tensor with shape (B, 11). All information necessary for sampling along a ray, including:
                     ray origin, ray direction, min dist, max dist, and unit-magnitude viewing direction.
        :param n_samples_coarse: Number of different times to sample along each ray for coarse phase.
        :param perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.
        :param n_samples_fine: Number of different times to sample along each ray for refinement phase.

        :return:
            - rgb_map_coarse: torch.Tensor (B, 3)
            - disp_map_coarse: torch.Tensor (B, )
            - acc_map_coarse: torch.Tensor (B, )
            - weights_coarse: torch.Tensor (B, n_samples_coarse)
            - depth_map_coarse: torch.Tensor (B, 3)

            - rgb_map_fine: torch.Tensor (B, 3), Estimated RGB color of a ray. Comes from fine model.
            - disp_map_fine: torch.Tensor (B, ), Disparity map. 1 / depth.
            - acc_map_fine: torch.Tensor (B, ), Accumulated opacity along each ray. Comes from fine model.
            - weights_fine: torch.Tensor (B, n_samples_fine)
            - depth_map_fine: torch.Tensor (B, 3)

            - depth_std: torch.Tensor (B, ), Standard deviation of distances along ray for each sample.
        """
        batch_size = rays.shape[0]

        # rays_o, rays_d, rays_d_normalized are all with shape (B, 3)
        rays_o, rays_d, rays_d_normalized = rays[:, 0:3], rays[:, 3:6], rays[:, -3:]

        # bounds.shape = (B, 1, 2)
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])

        # near and far are both with shape (B, 1)
        near, far = bounds[..., 0], bounds[..., 1]

        # t_vals.shape = (n_samples_coarse, )
        t_vals = torch.linspace(0.0, 1.0, steps=n_samples_coarse).to(self._device)

        # depth.shape = (B, n_samples_coarse), depth[i] equals depth[j] for i, j in {0, 1, ..., B-1}
        depth = near + (far - near) * t_vals  # sample linearly in depth.

        if perturb > 0.0:
            mids = 0.5 * (depth[..., 1:] + depth[..., :-1])  # (B, n_samples_coarse - 1)
            upper = torch.cat([mids, depth[..., -1:]], -1)  # (B, n_samples_coarse)
            lower = torch.cat([depth[..., :1], mids], -1)  # (B, n_samples_coarse)

            # stratified samples in those intervals, t_rand.shape = (B, n_samples_coarse)
            t_rand = torch.rand(depth.shape).to(self._device)

            # Now depth is with noise, its shape is (B, n_samples_coarse)
            depth = lower + (upper - lower) * t_rand

        # points_for_coarse is with shape (B, n_samples_coarse, 3)
        points_for_coarse = (
            rays_o[..., None, :] + rays_d[..., None, :] * depth[..., :, None]
        )

        # Coarse NeRF inference.
        # color_rgb.shape = (B * n_samples_coarse, 3)
        # sigma.shape = (B * n_samples_coarse, 1)
        color_rgb_coarse, sigma_coarse = self._coarse_model(
            torch.reshape(
                points_for_coarse, shape=(-1, 3)
            ),  # (B * n_samples_coarse, 3)
            torch.reshape(
                rays_d_normalized[:, None, :].repeat(1, n_samples_coarse, 1),
                shape=(-1, 3),
            ),
        )

        color_rgb_coarse = torch.reshape(
            color_rgb_coarse, shape=(batch_size, n_samples_coarse, 3)
        )
        sigma_coarse = torch.reshape(
            sigma_coarse, shape=(batch_size, n_samples_coarse, 1)
        )

        # rgb_map_coarse.shape = (B, 3)
        # disp_map_coarse.shape = (B, )
        # acc_map_coarse.shape = (B, )
        # weights_coarse.shape = (B, n_samples_coarse)
        # depth_map_coarse.shape = (B, )
        (
            rgb_map_coarse,
            disp_map_coarse,
            acc_map_coarse,
            weights_coarse,
            depth_map_coarse,
        ) = self.decode(
            color_rgb=color_rgb_coarse,  # (B, n_samples_coarse, 3)
            sigma=sigma_coarse,  # (B, n_samples_coarse, 1)
            depth=depth,  # (B, n_samples_coarse)
            rays_d=rays_d,  # (B, 3)
        )

        ret = {
            "rgb_map_coarse": rgb_map_coarse,
            "disp_map_coarse": disp_map_coarse,
            "acc_map_coarse": acc_map_coarse,
            "weights_coarse": weights_coarse,
            "depth_map_coarse": depth_map_coarse,
        }

        # See Paper Section 5.2: Hierarchical volume sampling
        #
        # Instead of just using a single network to represent the scene, we simultaneously optimize two
        # networks: one “coarse” and one “fine”. We first sample a set of N_c (i.e., n_samples_coarse)
        # locations using stratified sampling, and evaluate the “coarse” network at these locations as
        # described in Eqns. 2 and 3. Given the output of this “coarse” network, we then produce a more
        # informed sampling of points along each ray where samples are biased towards the relevant parts
        # of the volume.
        if n_samples_fine > 0:
            # depth_mid.shape = (B, n_samples_coarse - 1)
            depth_mid = 0.50 * (depth[..., 1:] + depth[..., :-1])

            # depth.depth_samples = (B, n_samples_fine)
            depth_samples = self.sample_pdf(
                depth_mid,  # (B, n_samples_coarse - 1)
                weights_coarse[..., 1:-1],  # (B, n_samples_coarse - 2)
                n_samples_fine,
                det=(perturb == 0.0),
            )
            depth_samples = depth_samples.detach()

            # depth.shape = (B, n_samples_coarse + n_samples_fine)
            depth, _ = torch.sort(torch.cat([depth, depth_samples], dim=-1), dim=-1)

            # points_for_refinement.shape = [B, n_samples_coarse + n_samples_fine, 3]
            points_for_refinement = (
                rays_o[..., None, :] + rays_d[..., None, :] * depth[..., :, None]
            )

            # Fine NeRF inference.
            # color_rgb.shape = (B * n_samples_coarse, 3)
            # sigma.shape = (B * n_samples_coarse, 1)
            color_rgb_fine, sigma_fine = self._fine_model(
                torch.reshape(
                    points_for_refinement, shape=(-1, 3)
                ),  # (B * (n_samples_coarse + n_samples_fine), 3)
                torch.reshape(
                    rays_d_normalized[:, None, :].repeat(
                        1, n_samples_coarse + n_samples_fine, 1
                    ),
                    shape=(-1, 3),
                ),
            )

            color_rgb_fine = torch.reshape(
                color_rgb_fine, shape=(batch_size, n_samples_coarse + n_samples_fine, 3)
            )
            sigma_fine = torch.reshape(
                sigma_fine, shape=(batch_size, n_samples_coarse + n_samples_fine, 1)
            )

            # rgb_map_fine.shape = (B, 3)
            # disp_map_fine.shape = (B, )
            # acc_map_fine.shape = (B, )
            # weights_fine.shape = (B, n_samples_coarse + n_samples_fine)
            # depth_map_fine.shape = (B, )
            rgb_map_fine, disp_map_fine, acc_map_fine, weights_fine, depth_map_fine = (
                self.decode(
                    color_rgb=color_rgb_fine,  # (B, n_samples_coarse + n_samples_fine, 3)
                    sigma=sigma_fine,  # (B, n_samples_coarse + n_samples_fine, 1)
                    depth=depth,  # (B, n_samples_coarse + n_samples_fine)
                    rays_d=rays_d,  # (B, 3)
                )
            )

            ret["rgb_map_fine"] = rgb_map_fine
            ret["disp_map_fine"] = disp_map_fine
            ret["acc_map_fine"] = acc_map_fine
            ret["weights_fine"] = weights_fine
            ret["depth_map_fine"] = depth_map_fine
            ret["depth_std"] = torch.std(depth_samples, dim=-1, unbiased=False)

        return ret

    def render(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float = 2.0,
        far: float = 6.0,
    ):
        """
        Volumetric rendering.

        :param rays_o: rays' origins, shape is (B, 3).
        :param rays_d: rays' direction vectors, shape is (B, 3).
        :param near: bounded space's minimum distance, defaults to 2.0.
        :param far: bounded space's maximum distance, defaults to 6.0.

        :return:
            rgb_map: torch.Tensor with shape (B, 3), i.e., predicted RGB values for rays.
            disp_map: torch.Tensor with shape (B, ), i.e., disparity map, inverse of depth.
            acc_map: torch.Tensor with shape (B, ), i.e., accumulated opacity (alpha) along a ray.
            extras: dictionary with everything returned by render_rays() function.
        """
        rays_d_normalized = rays_d

        # rays_d_normalized.shape = (B, 3)
        rays_d_normalized = rays_d_normalized / torch.norm(
            rays_d_normalized, dim=-1, keepdim=True
        )

        near = near * torch.ones_like(rays_d[..., :1])  # near.shape = (B, 1)
        far = far * torch.ones_like(rays_d[..., :1])  # far.shape = (B, 1)
        rays = torch.cat(
            [rays_o, rays_d, near, far, rays_d_normalized], dim=-1
        )  # rays.shape = (B, 11)

        response = self.render_rays(
            rays=rays,
            n_samples_coarse=self._n_samples_coarse,
            perturb=self._perturb,
            n_samples_fine=self._n_samples_fine,
        )

        return response

    def run(self):
        with torch.no_grad():
            # For validation and test phase, the total samples are 100, 200, respectively.
            # The rays for validation dataset is 100 * 400 * 400 = 16000000
            # The rays for test dataset is 200 * 400 * 400 = 32000000

            # predicted_rendering_images.shape = (N, 400, 400)
            predicted_rendering_images = torch.zeros(
                size=(
                    self._val_data.images_amount,
                    self._val_data.image_height,
                    self._val_data.image_width,
                    3,
                )
            )

            # gt_images.shape = (N, 400, 400)
            gt_images = torch.zeros(
                size=(
                    self._val_data.images_amount,
                    self._val_data.image_height,
                    self._val_data.image_width,
                    3,
                )
            )

            predicted_rendering_images = torch.reshape(
                predicted_rendering_images, shape=(-1, 3)
            )
            gt_images = torch.reshape(gt_images, shape=(-1, 3))

            for batch_index, data in enumerate(self._val_loader):
                # Step 1: Parse data
                #
                # batch_rays_with_direction.shape = (B, 2, 3)
                # batch_target_rgb.shape = (B, 3)
                (batch_rays_with_direction, batch_target_rgb) = data
                batch_rays_with_direction = batch_rays_with_direction.to(self._device)
                batch_target_rgb = batch_target_rgb.to(self._device)

                # Step 2: Volume rendering (NeRF inference inside)
                prediction = self.render(
                    rays_o=batch_rays_with_direction[:, 0, :],
                    rays_d=batch_rays_with_direction[:, 1, :],
                    near=self._distance_min,
                    far=self._distance_max,
                )

                rgb_map_coarse = prediction["rgb_map_coarse"]  # (B, 3)
                disp_map_coarse = prediction["disp_map_coarse"]  # (B, )
                acc_map_coarse = prediction["acc_map_coarse"]  # (B, )
                weights_coarse = prediction["weights_coarse"]  # (B, n_samples_coarse)
                depth_map_coarse = prediction["depth_map_coarse"]  # (B, )

                rgb_map_fine = prediction["rgb_map_fine"]  # (B, 3)
                disp_map_fine = prediction["disp_map_fine"]  # (B, )
                acc_map_fine = prediction["acc_map_fine"]  # (B, )
                weights_fine = prediction[
                    "weights_fine"
                ]  # (B, n_samples_coarse + n_samples_fine)
                depth_map_fine = prediction["depth_map_fine"]  # (B, )

                depth_std = prediction["depth_std"]

                # Collect prediction & ground truth RGB values for later visualization.
                predicted_rendering_images[
                    batch_index
                    * self._batch_size : (batch_index + 1)
                    * self._batch_size,
                    :,
                ] = rgb_map_fine
                gt_images[
                    batch_index
                    * self._batch_size : (batch_index + 1)
                    * self._batch_size,
                    :,
                ] = batch_target_rgb

                # Step 3: Loss calculation.
                mse_loss = self._mse_loss(
                    rgb_map_coarse, batch_target_rgb
                ) + self._mse_loss(rgb_map_fine, batch_target_rgb)
                psnr = get_psnr(mse=mse_loss)

                print(
                    "Batch {}/{}: MSE Loss = {:.6f}, PSNR = {:.6f}".format(
                        self._max_epochs,
                        batch_index + 1,
                        len(self._val_loader),
                        mse_loss.item(),
                        psnr.item(),
                    )
                )

        # Visualization
        predicted_rendering_images = predicted_rendering_images.detach().cpu().numpy()
        gt_images = gt_images.detach().cpu().numpy()

        for i in range(self._val_data.images_amount):
            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(predicted_rendering_images[i])

            plt.subplot(1, 2, 2)
            plt.imshow(gt_images[i])

            plt.show()


def main():
    task = Predictor(
        config={
            "MinDistance": 2.0,
            "MaxDistance": 6.0,
            "CoarseModelFullPath": "./runs/nerf_coarse_epoch_10_batch_15625.pth",
            "FineModelFullPath": "./runs/nerf_fine_epoch_10_batch_15625.pth",
            "NOfCoarse": 64,
            "Perturb": 1.0,
            "NOfFine": 128,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
