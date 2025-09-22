# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union
import torch


# Refer to https://github.com/Angtian/VoGE/blob/main/VoGE/Utils.py
def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int = 1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :return: sel_target [... (k), M, ...]
    """
    assert (
        len(ind.shape) > dim
    ), "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(
        *tuple(
            [ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)]
            + [
                -1,
            ]
            * (len(target.shape) - dim)
        )
    )

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1) : :])

    return torch.gather(target, dim=dim, index=ind_pad)


def merge_final(vert_attr: torch.Tensor, weight: torch.Tensor, vert_assign: torch.Tensor):
    """

    :param vert_attr: [n, d] or [b, n, d] color or feature of each vertex
    :param weight: [b(optional), w, h, M] weight of selected vertices
    :param vert_assign: [b(optional), w, h, M] selective index
    :return:
    """
    target_dim = len(vert_assign.shape) - 1
    if len(vert_attr.shape) == 2:
        assert vert_attr.shape[0] > vert_assign.max()
        # [n, d] ind: [b(optional), w, h, M]-> [b(optional), w, h, M, d]
        # sel_attr = ind_sel(
        #     vert_attr[(None,) * target_dim], vert_assign.type(torch.long), dim=target_dim
        # )
        new_shape = [1] * target_dim + list(vert_attr.shape)
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.type(torch.long), dim=target_dim)
    else:
        assert vert_attr.shape[1] > vert_assign.max()
        #sel_attr = ind_sel(
        #    vert_attr[:, *(None,) * (target_dim - 1)], vert_assign.type(torch.long), dim=target_dim
        #)
        new_shape = [vert_attr.shape[0]] + [1] * (target_dim - 1) + list(vert_attr.shape[1:])
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.type(torch.long), dim=target_dim)

    # [b(optional), w, h, M]
    final_attr = torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)
    return final_attr


def patch_motion(tracks, vid, topk=2, temperature=25.0, vae_divide=(16,)):
    """
    tracks: (B, T, N, 4)  where last dim = [mask(1), x(1), y(1), visible(1)]
            (some builds use [mask, x, y, visible], some [dummy, x, y, visible])
    vid:    (C, T, H, W)
    Returns: (C + vae_divide[0], T, H, W)   # mask channels prepended as in your pipeline
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        C, T, H, W = vid.shape
        B = tracks.shape[0]
        N = tracks.shape[2]

        # Split to (B, T, N, 2) coords and (B, T, N, 1) visibility
        # Your tracks have 4 at the end; we keep the middle two as (x,y) and last as visible.
        _, xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)  # xy: (B,T,N,2) | visible: (B,T,N,1)

        # Normalize coords to [-1,1] in the same way your original code did
        s = float(min(H, W))
        norm = torch.tensor([W / s, H / s], device=tracks.device, dtype=vid.dtype)
        xy_n = (xy / norm).clamp(-1, 1)  # (B,T,N,2)
        visible = visible.clamp(0, 1)    # (B,T,N,1)

        # Build grid (H,W,2) on the right device/dtype
        xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
        yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)

        # Drop the first frame for alignment against (T-1)
        xy_pad = xy[:, 1:]           # (B, T-1, N, 2)
        vis_pad = visible[:, 1:]     # (B, T-1, N, 1)
        Tm1 = vis_pad.shape[1]       # <-- correct time length (T-1)
        Tcalc = Tm1 + 1              # just for clarity

        # Reduce over batch (works for B==1 and B>1)
        if B == 1:
            vis_sum = vis_pad.squeeze(0)                 # (T-1, N, 1)
            xy_sum  = (xy_pad.squeeze(0) * vis_sum)      # (T-1, N, 2)
        else:
            vis_sum = vis_pad.sum(0)                     # (T-1, N, 1)
            xy_sum  = (xy_pad * vis_pad).sum(0)          # (T-1, N, 2)

        # Weighted mean positions where visible>0
        eps = 1e-5
        align_vis = vis_sum                               # (T-1, N, 1)
        align_xy  = xy_sum / (align_vis + eps)            # (T-1, N, 2)

        # Distance field to each track point (broadcast to (T-1,H,W,N))
        # (T-1,N,2) vs (H,W,2) -> (T-1,H,W,N,2) -> sum over last
        diff = align_xy[:, None, None, :, :] - grid[None, :, :, None, :]   # (T-1,H,W,N,2)
        dist = (diff * diff).sum(-1)                                       # (T-1,H,W,N)

        # Convert to weights and keep only visible tracks
        vis_mask = align_vis.squeeze(-1)                    # (T-1, N)
        weight = torch.exp(-dist * temperature) * vis_mask[:, None, None, :]  # (T-1,H,W,N)

        # Top-k over tracks (N)
        k = min(int(topk), weight.shape[-1])
        vert_weight, vert_index = torch.topk(weight, k=k, dim=-1)           # (T-1,H,W,k), (T-1,H,W,k)

    # Sample a per-track feature at the FIRST time step (kept to match your current behavior)
    grid_mode = "bilinear"
    # vid: (C,T,H,W) -> (T,C,H,W) to grid_sample per time
    # You only use frame 0 features for all tracks; leaving this identical to your pipeline.
    point_feature = F.grid_sample(
        vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1],       # (1,C,H,W)
        xy_n[:, :1].type(vid.dtype),                       # (B,1,N,2) -> treated as (N,2) per grid_sample batching
        mode=grid_mode,
        padding_mode="zeros",
        align_corners=False,
    )
    # -> (1,C,1,N)  squeeze to (N,C)
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0)       # (N, C)

    # Merge back to per-pixel features (your existing helper)
    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2)  # (C,T-1,H,W)
    out_weight  = vert_weight.sum(-1)                                                     # (T-1,H,W)

    # Soft blend with original latent features (unchanged behavior)
    mix_feature = out_feature + vid[vae_divide[0]:, 1:] * (1 - out_weight.clamp(0, 1))   # (C,T-1,H,W)

    # Re-attach the first frame
    out_feature_full = torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1)          # (C,T,H,W)
    out_mask_full    = torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0)   # (T,H,W)

    # Prepend mask channels for the VAE split as your pipeline expects
    return torch.cat(
        [out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full],
        dim=0
    )