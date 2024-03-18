#!/usr/bin/python3
#
# ------------------------------------------------------------------------------
# Author: Gaowei Xu (gaowexu1991@gmail.com)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn


class NeRF(nn.Module):
    def __init__(
        self,
        embedding_dim_pos: int = 10,
        embedding_dim_direction: int = 4,
        hidden_dim: int = 256,
    ):
        super(NeRF, self).__init__()

        self._embedding_dim_pos = embedding_dim_pos
        self._embedding_dim_direction = embedding_dim_direction
        self._hidden_dim = hidden_dim

        self._block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self._block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )

        self._block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + 3 + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # RGB regression: output channels is 3, which indicates RGB channels.
        self._block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        self._relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x: torch.Tensor, L: int = 10):
        """
        Positional encoding.

        :param x: torch.Tensor with shape (B, 3), 3 indicates (x, y, z) for camera origin or (dx, dy dz) for ray
                  direction vector.
        :param L: parameter for encoded positional information. L = 10 for camera origin positional encoding, L = 4 for
                  ray direction vector positional encoding.

        :return: torch.Tensor with shape (B, 3 * 2 * L + 3)
        """
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))

        return torch.cat(out, dim=1)

    def forward(self, o: torch.Tensor, d: torch.Tensor):
        """
        NeRF forward function.

        :param o: torch.Tensor with shape (B, 3), 3 indicates (x, y, z) for camera origin.
        :param d: torch.Tensor with shape (B, 3), 3 indicates (dx, dy, dz) for ray direction vector.

        :return:
        """
        # emb_x.shape = (B, 3 * 2 * L + 3), L = 10
        emb_x = self.positional_encoding(o, self._embedding_dim_pos)

        # emb_d.shape = (B, 3 * 2 * L + 3), L = 4
        emb_d = self.positional_encoding(d, self._embedding_dim_direction)

        # h.shape = (B, hidden_dim), i.e., (B, 256)
        h = self._block1(emb_x)

        # (B, 256) + (B, 63) --> (B, 319)
        feat = torch.cat((h, emb_x), dim=1)
        encoded_feat = self._block2(feat)

        h = encoded_feat[:, :-1]
        sigma = self._relu(encoded_feat[:, -1])

        # (B, 256) + (B, 27) --> (B,283)
        rgb_feat = torch.cat((h, emb_d), dim=1)

        h = self._block3(rgb_feat)
        color_rgb = self._block4(h)

        return color_rgb, sigma


if __name__ == "__main__":
    B = 1024

    rays_o = torch.rand(B, 3)
    rays_d = torch.rand(B, 3)

    model = NeRF()

    color_rgb, sigma = model(rays_o, rays_d)

    print("color_rgb.shape = {}".format(color_rgb.shape))
    print("sigma.shape = {}".format(sigma.shape))