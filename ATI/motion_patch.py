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
    Robust ATI motion patch that tolerates sampler/ATI step mismatches.

    tracks: (B, T, N, 4)  where last dim = [mask_or_dummy(1), x(1), y(1), visible(1)]
            (your host uses [mask,dummy] or [mask], we only consume x,y,visible)
    vid:    (C, T, H, W)   latent video features
    Returns:
      (C + vae_divide[0], T, H, W) with mask channels prepended as your pipeline expects.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        # ---------------------------
        # Shapes, splits, normalization
        # ---------------------------
        C, T, H, W = vid.shape
        B = tracks.shape[0]
        N = tracks.shape[2]

        # Take only x,y,visible from the last dim
        # (_, xy (2), visible (1))
        _, xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)  # xy: (B,T,N,2), visible: (B,T,N,1)

        # Normalize XY to [-1, 1] w.r.t. min(H,W) like the original code path
        s = float(min(H, W))
        norm = torch.tensor([W / s, H / s], device=vid.device, dtype=vid.dtype)
        xy_n = (xy.to(vid.dtype) / norm).clamp(-1, 1)           # (B,T,N,2)
        visible = visible.clamp(0, 1)                           # (B,T,N,1)

        # Build grid (H,W,2) for distance field (same device/dtype as vid)
        xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
        yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)

        # -----------------------------------
        # Align to (T-1) and tolerate mismatch
        # -----------------------------------
        # We align to motion *between* frames, so drop the first time step
        xy_pad = xy[:, 1:]            # (B, T-1, N, 2)
        vis_pad = visible[:, 1:]      # (B, T-1, N, 1)

        # Sampler-side conditional steps (T-1)
        steps_cond = max(T - 1, 0)
        # ATI-side steps present in vis_pad (second dim)
        steps_tracks = vis_pad.shape[1]

        # Use the minimum, slice both tensors to match
        steps = min(steps_cond, steps_tracks)
        if steps <= 0:
            # No motion steps — return original with zero mask channels
            zero_mask = torch.zeros((vae_divide[0], T, H, W), device=vid.device, dtype=vid.dtype)
            return torch.cat([zero_mask, vid], dim=0)

        xy_pad = xy_pad[:, :steps]       # (B, steps, N, 2)
        vis_pad = vis_pad[:, :steps]     # (B, steps, N, 1)

        # -----------------------------------
        # Reduce across batch: weighted mean xy per (t, n)
        # -----------------------------------
        # Sum visible and (xy * visible) across batch
        # vis_sum: (steps, N, 1), xy_sum: (steps, N, 2)
        if B == 1:
            vis_sum = vis_pad.squeeze(0)                  # (steps, N, 1)
            xy_sum  = (xy_pad.squeeze(0) * vis_sum)       # (steps, N, 2)
        else:
            vis_sum = vis_pad.sum(0)                      # (steps, N, 1)
            xy_sum  = (xy_pad * vis_pad).sum(0)           # (steps, N, 2)

        eps = 1e-5
        align_vis = vis_sum                                # (steps, N, 1)
        align_xy  = xy_sum / (align_vis + eps)             # (steps, N, 2)

        # -----------------------------------
        # Distance -> weights, top-k per pixel
        # -----------------------------------
        # dist: (steps, H, W, N)
        diff = align_xy[:, None, None, :, :] - grid[None, :, :, None, :]   # (steps,H,W,N,2)
        dist = (diff * diff).sum(-1)                                       # (steps,H,W,N)

        # Apply visibility mask to weights
        vis_mask = align_vis.squeeze(-1)                   # (steps, N)
        weight = torch.exp(-dist * float(temperature)) * vis_mask[:, None, None, :]  # (steps,H,W,N)

        k = int(min(max(topk, 1), max(1, weight.shape[-1])))  # clamp k
        vert_weight, vert_index = torch.topk(weight, k=k, dim=-1)          # (steps,H,W,k), (steps,H,W,k)

    # -----------------------------------
    # Sample per-track point feature from frame 0 (current behavior)
    # -----------------------------------
    grid_mode = "bilinear"

    # Make a grid for grid_sample with shape (N, 1, 2) per PyTorch convention:
    # We treat H_out=1, W_out=N by using a single-batch 1xN grid.
    # For B>1 we take the first batch’s points (consistent with prior behavior).
    xy0 = xy_n[0:1, :1]                                   # (1, 1, N, 2)
    # Reformat to grid (1, 1, N, 2) -> (1, 1, N, 2) works directly with (1,C,H,W)
    point_feature = F.grid_sample(
        vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1],      # (1, C, H, W) — first frame features
        xy0.to(vid.dtype),                                # (1, 1, N, 2)
        mode=grid_mode,
        padding_mode="zeros",
        align_corners=False,
    )
    # (1, C, 1, N) -> (N, C)
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0).contiguous()  # (N, C)

    # -----------------------------------
    # Merge back to per-pixel features via your helper
    # -----------------------------------
    # merge_final(point_feature, vert_weight, vert_index) is assumed to exist,
    # returning (H, W, steps, C) or compatible. We’ll reorder to (C, steps, H, W).
    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 2, 0, 1)  # (C, steps, H, W)

    # Aggregate weights for soft blend (steps, H, W)
    out_weight = vert_weight.sum(-1).clamp(0, 1)          # (steps, H, W)

    # -----------------------------------
    # Blend with original latents and restore first frame
    # -----------------------------------
    # Ensure vid has at least (steps+1) frames (it does; we used steps <= T-1)
    base_tail = vid[vae_divide[0]:, 1:1+steps]            # (C, steps, H, W)
    mix_feature = out_feature + base_tail * (1 - out_weight[None])

    # Rebuild full sequence: first frame + mixed tail
    out_feature_full = torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1)   # (C, T, H, W)

    # Build mask channels: one zero/one mask for t=0, then weights for following frames
    # If steps < T-1 (we trimmed), pad remaining masks with zeros to keep length T
    mask_head = torch.ones_like(out_weight[:1])                              # (1, H, W)
    if steps < (T - 1):
        pad_len = (T - 1) - steps
        pad_tail = torch.zeros((pad_len, H, W), device=vid.device, dtype=vid.dtype)
        out_mask_full = torch.cat([mask_head, out_weight, pad_tail], dim=0)  # (T, H, W)
    else:
        out_mask_full = torch.cat([mask_head, out_weight], dim=0)            # (T, H, W)

    # Prepend mask channels for VAE split
    mask_stack = out_mask_full[None].expand(vae_divide[0], -1, -1, -1)       # (vae_divide[0], T, H, W)
    return torch.cat([mask_stack, out_feature_full], dim=0)
