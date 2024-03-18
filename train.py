#!/usr/bin/python3
#
# ------------------------------------------------------------------------------
# Author: Gaowei Xu (gaowexu1991@gmail.com)
# ------------------------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
import numpy as np
from nerf import NeRF
import torch.optim as optim
from data_loader import compile_data
from loss import get_mse_loss, get_psnr


class TrainTask(object):
    def __init__(self, config: dict):
        self._config = config

        self._max_epochs = self._config["MaxEpochs"]
        self._batch_size = self._config["BatchSize"]
        self._initialized_lr = self._config["InitialLearningRate"]
        self._target_lr = self._config["TargetLearningRate"]
        self._warmup_epochs = self._config["WarmupEpochs"]
        self._div_factor = self._config["DivFactor"]
        self._decay_steps = self._config["DecaySteps"]
        self._models_root_dir = self._config["ModelZooDir"]
        self._distance_min = self._config["MinDistance"]
        self._distance_max = self._config["MaxDistance"]

        if not os.path.exists(self._models_root_dir):
            os.makedirs(self._models_root_dir)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = NeRF()
        self._model = self._model.to(self._device)

        self._train_loader, self._val_loader, self._test_loader = compile_data(
            batch_size=self._batch_size, num_workers=16
        )

        self._batch_total_num = len(self._train_loader) * self._max_epochs
        self._optimizer = optim.AdamW(
            self._model.parameters(), lr=self._target_lr, weight_decay=0.01, eps=1e-3
        )

        # Create learning rate scheduler
        self._lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=self._optimizer,
            lr_lambda=lambda epoch: (
                (
                    (self._target_lr - self._initialized_lr)
                    * epoch
                    / self._warmup_epochs
                    + self._initialized_lr
                )
                / self._target_lr
                if epoch <= self._warmup_epochs
                else self._div_factor
                ** len([m for m in self._decay_steps if m <= epoch])
            ),
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
            [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], dim=-1
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
                    [torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], dim=-1
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
    def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0.0, 1.0, steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0.0, 1.0, N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
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
        n_samples: int = 64,
        perturb: float = 1.0,
        n_importance: int = 128,
        # network_fn,
        # network_query_fn,
        # N_samples,
        # retraw=False,
        # lindisp=False,
        # perturb=0.0,
        # N_importance=0,
        # network_fine=None,
        # white_bkgd=False,
        # raw_noise_std=0.0,
        # verbose=False,
        # pytest=False,
    ):
        """Volumetric rendering.
        Args:
        rays: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
        network_fn: function. Model for predicting RGB and density at each point
            in space.
        network_query_fn: function used for passing queries to network_fn.
        N_samples: int. Number of different times to sample along each ray.
        retraw: bool. If True, include model's raw, unprocessed predictions.
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
        N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
        network_fine: "fine" network with same spec as network_fn.
        white_bkgd: bool. If True, assume a white background.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
        Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        batch_size = rays.shape[0]

        # rays_o, rays_d, rays_d_normalized are all with shape (B, 3)
        rays_o, rays_d, rays_d_normalized = rays[:, 0:3], rays[:, 3:6], rays[:, -3:]

        # bounds.shape = (B, 1, 2)
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])

        # near and far are both with shape (B, 1)
        near, far = bounds[..., 0], bounds[..., 1]

        # t_vals.shape = (n_samples, )
        t_vals = torch.linspace(0.0, 1.0, steps=n_samples)

        # depth.shape = (B, n_samples), depth[i] equals depth[j] for i, j in {0, 1, ..., B-1}
        depth = near + (far - near) * t_vals  # sample linearly in depth.

        if perturb > 0.0:
            mids = 0.5 * (depth[..., 1:] + depth[..., :-1])  # (B, n_samples - 1)
            upper = torch.cat([mids, depth[..., -1:]], -1)  # (B, n_samples)
            lower = torch.cat([depth[..., :1], mids], -1)  # (B, n_samples)

            # stratified samples in those intervals, t_rand.shape = (B, n_samples)
            t_rand = torch.rand(depth.shape)

            # Now depth is with noise, its shape is (B, n_samples)
            depth = lower + (upper - lower) * t_rand

        # points is with shape (B, n_samples, 3)
        points = rays_o[..., None, :] + rays_d[..., None, :] * depth[..., :, None]

        # NeRF inference.
        # color_rgb.shape = (B * n_samples, 3)
        # sigma.shape = (B * n_samples, 1)
        color_rgb, sigma = self._model(
            torch.reshape(points, shape=(-1, 3)),  # (B * n_samples, 3)
            torch.reshape(
                rays_d_normalized[:, None, :].repeat(1, n_samples, 1), shape=(-1, 3)
            ),
        )

        color_rgb = torch.reshape(color_rgb, shape=(batch_size, n_samples, 3))
        sigma = torch.reshape(sigma, shape=(batch_size, n_samples, 1))

        # rgb_map.shape = (B, 3)
        # disp_map.shape = (B, )
        # acc_map.shape = (B, )
        # weights.shape = (B, n_samples)
        # depth_map.shape = (B, )
        rgb_map, disp_map, acc_map, weights, depth_map = self.decode(
            color_rgb=color_rgb,  # (B, n_samples, 3)
            sigma=sigma,  # (B, n_samples, 1)
            depth=depth,  # (B, n_samples)
            rays_d=rays_d,  # (B, 3)
        )

        if n_importance > 0:
            # depth_mid.shape = (B, n_samples - 1)
            depth_mid = 0.50 * (depth[..., 1:] + depth[..., :-1])

            # depth.depth_samples = (B, n_importance)
            depth_samples = self.sample_pdf(
                depth_mid,  # (B, n_samples - 1)
                weights[..., 1:-1],  # (B, n_samples - 2)
                n_importance,
                det=(perturb == 0.0),
                pytest=pytest,
            )
            depth_samples = depth_samples.detach()

            # depth.shape = (B, n_samples + n_importance)
            depth, _ = torch.sort(torch.cat([depth, depth_samples], dim=-1), dim=-1)

            # points_for_refinement.shape = [B, N_samples + N_importance, 3]
            points_for_refinement = (
                rays_o[..., None, :] + rays_d[..., None, :] * depth[..., :, None]
            )

            run_fn = network_fn if network_fine is None else network_fine
            raw = network_query_fn(points_for_refinement, rays_d_normalized, run_fn)

            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
                raw, depth, rays_d, raw_noise_std, white_bkgd, pytest=pytest
            )

        ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
        if retraw:
            ret["raw"] = raw

        if N_importance > 0:
            ret["rgb0"] = rgb_map_0
            ret["disp0"] = disp_map_0
            ret["acc0"] = acc_map_0
            ret["z_std"] = torch.std(depth_samples, dim=-1, unbiased=False)  # [B]

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

        sh = rays_d.shape

        near = near * torch.ones_like(rays_d[..., :1])  # near.shape = (B, 1)
        far = far * torch.ones_like(rays_d[..., :1])  # far.shape = (B, 1)
        rays = torch.cat(
            [rays_o, rays_d, near, far, rays_d_normalized], dim=-1
        )  # rays.shape = (B, 11)

        # Render & reshape.
        batch_size = rays.shape[0]

        all_ret = dict()
        response = self.render_rays(
            rays=rays,
            n_samples=self._n_samples,
            retraw=True,
        )

        for k in response:
            if k not in all_ret:
                all_ret[k] = list()

            all_ret[k].append(response[k])

        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ["rgb_map", "disp_map", "acc_map"]
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

        return ret_list + [ret_dict]

    def save_model(self, epoch_index: int, batch_index: int):
        self._model.eval()
        pth_model_full_path = os.path.join(
            self._models_root_dir,
            "nerf_epoch_{}_batch_{}.pth".format(epoch_index, batch_index),
        )
        torch.save(
            self._model,
            pth_model_full_path,
        )
        self._model.train()
        return

    def run(self):
        self._model.train()
        for epoch_index in range(self._max_epochs):
            for batch_index, data in enumerate(self._train_loader):
                # Step 1: Parse data
                #
                # batch_rays_with_direction.shape = (B, 2, 3)
                # batch_target_rgb.shape = (B, 3)
                (batch_rays_with_direction, batch_target_rgb) = data
                batch_rays_with_direction = batch_rays_with_direction.to(self._device)
                batch_target_rgb = batch_target_rgb.to(self._device)

                # Step 2: Volume rendering (NeRF inference inside)
                pred_rays_rgb, disparity_map, accumulated_opacity, extras = self.render(
                    rays_o=batch_rays_with_direction[:, 0, :],
                    rays_d=batch_rays_with_direction[:, 1, :],
                    near=self._distance_min,
                    far=self._distance_max,
                )

                # Step 3: Loss calculation.
                mse_loss = self._mse_loss(pred_rays_rgb, batch_target_rgb)
                psnr = get_psnr(mse=mse_loss)

                print(
                    "Epoch {}/{}, Batch {}/{}: Learning Rate = {:.6f}, MSE Loss = {:.6f}, PSNR = {:.6f}".format(
                        epoch_index + 1,
                        self._max_epochs,
                        batch_index + 1,
                        len(self._train_loader),
                        self._optimizer.param_groups[0]["lr"],
                        mse_loss.item(),
                        psnr.item(),
                    )
                )

                # Step 4: Loss back-propagation.
                self._optimizer.zero_grad()
                mse_loss.backward()
                self._optimizer.step()

                # Step 5: Save model artifacts and weights.
                if batch_index == len(self._train_loader) - 1:
                    self.save_model(epoch_index + 1, batch_index + 1)

            # Multi-stage lambda learning rate scheduler.
            self._lr_scheduler.step()


def main():
    task = TrainTask(
        config={
            "MaxEpochs": 10,
            "BatchSize": 4096,
            "InitialLearningRate": 0.0001,
            "TargetLearningRate": 0.001,
            "WarmupEpochs": 5,
            "DivFactor": 0.10,
            "DecaySteps": [30, 40],
            "MinDistance": 2.0,
            "MaxDistance": 6.0,
            "ModelZooDir": "./runs",
        }
    )
    task.run()


if __name__ == "__main__":
    main()
