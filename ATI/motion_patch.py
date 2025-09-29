print(">>> USING motion_patch FROM:", __file__)

import torch
import torch.nn.functional as F

def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int = 1):
    assert (len(ind.shape) > dim), "Index must have the target dim"
    target = target.expand(
        *tuple(
            [ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)]
            + [-1] * (len(target.shape) - dim)
        )
    )
    ind_pad = ind
    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1):])
    return torch.gather(target, dim=dim, index=ind_pad)

def merge_final(vert_attr: torch.Tensor, weight: torch.Tensor, vert_assign: torch.Tensor):
    target_dim = len(vert_assign.shape) - 1
    if len(vert_attr.shape) == 2:
        new_shape = [1] * target_dim + list(vert_attr.shape)
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.long(), dim=target_dim)
    else:
        new_shape = [vert_attr.shape[0]] + [1] * (target_dim - 1) + list(vert_attr.shape[1:])
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.long(), dim=target_dim)
    return torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)

def _resample_tracks_time(tracks: torch.Tensor, T_target: int) -> torch.Tensor:
    """
    tracks: (B, T_src, N, 4) with [mask_or_dummy, x, y, visible]
    returns: (B, T_target, N, 4) with time-resampled x,y,visible and mask=1
    """
    B, T_src, N, D = tracks.shape
    assert D == 4, f"expected last dim 4, got {D}"
    device, dtype = tracks.device, tracks.dtype

    # Split off xyv (drop mask/dummy)
    _, xyv = torch.split(tracks, [1, 3], dim=-1)        # (B, T_src, N, 3)

    # Prepare for 1D linear interpolation over time
    # (B, T_src, N, 3) -> (B, N, 3, T_src) -> (B*N, 3, T_src)
    x = xyv.permute(0, 2, 3, 1).reshape(B * N, 3, T_src)

    # Interpolate to target T
    x = F.interpolate(x, size=T_target, mode="linear", align_corners=False)  # (B*N, 3, T_target)

    # Back to (B, T_target, N, 3)
    xyv_new = x.reshape(B, N, 3, T_target).permute(0, 3, 1, 2).contiguous()

    # Rebuild mask channel as ones (or keep your own rule if needed)
    mask = torch.ones((B, T_target, N, 1), device=device, dtype=dtype)

    return torch.cat([mask, xyv_new], dim=-1)           # (B, T_target, N, 4)


@torch.inference_mode()
def _weighted_gather_fuse(point_feature, vert_weight, vert_index):
    """
    point_feature: (N, C)
    vert_weight :  (T-1, H, W, K)
    vert_index  :  (T-1, H, W, K)  indices into N
    returns     :  (C, T-1, H, W)
    """
    # Gather per-pixel top-K features -> (T-1, H, W, K, C)
    # Use advanced indexing instead of any reshape/view gymnastics.
    gathered = point_feature[vert_index]                             # (T-1, H, W, K, C)

    # Normalize weights across K, then weighted sum over K
    w = vert_weight
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)              # (T-1, H, W, 1)
    w_norm = w / w_sum                                                # (T-1, H, W, K)
    fused = (gathered * w_norm[..., None]).sum(dim=-2)                # (T-1, H, W, C)

    # -> (C, T-1, H, W)
    return fused.permute(3, 0, 1, 2).contiguous()

@torch.inference_mode()
def patch_motion(
    tracks: torch.FloatTensor,  # (B, T, N, 4) -> [mask/dummy, x, y, visible]
    vid: torch.FloatTensor,     # (C, T, H, W)
    temperature: float = 220.0,
    vae_divide: tuple = (4, 16),
    topk: int = 2,
):
    with torch.no_grad():
        C, T, H, W = vid.shape
        B, _, N, _ = tracks.shape

        # Guard: Wan latent time expects (T-1) divisible by 4 (e.g., 81 -> 21 -> 20)
        if (T - 1) % 4 != 0:
            raise ValueError(f"(T-1) must be divisible by 4; got T={T} from vid.shape. Check num_frames (should be 4k+1).")

        # Split xy + visibility and normalize xy into [-1,1] in pixel space
        _, xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)   # (B,T,N,2), (B,T,N,1)
        s = float(min(H, W))
        norm = torch.tensor([W / s, H / s], device=vid.device, dtype=vid.dtype)
        xy_n = (xy / norm).clamp(-1, 1)                           # (B,T,N,2)
        visible = visible.clamp(0, 1)                             # (B,T,N,1)

        # Static grid in pixel coords
        xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
        yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)

        # Align to T-1 (next frame), then reduce over batch first (robust for any B)
        xy_pad = xy[:, 1:]        # (B,T-1,N,2)
        vis_pad = visible[:, 1:]  # (B,T-1,N,1)

        # Safe batch reduction (keeps original math when B==1)
        vis_sum = vis_pad.sum(0)              # (T-1,N,1)
        xy_sum  = (xy_pad * vis_pad).sum(0)   # (T-1,N,2)

        # Now reshape time into (groups of 4) without touching batch
        t_groups = (T - 1) // 4
        vis_blocks = vis_sum.reshape(t_groups, 4, N, 1).sum(1)           # (t_groups, N, 1)
        xy_blocks  = xy_sum.reshape(t_groups, 4, N, 2).sum(1)            # (t_groups, N, 2)
        eps = 1e-5
        align_vis = vis_blocks                                           # (t_groups,N,1)
        align_xy  = xy_blocks / (align_vis + eps)                        # (t_groups,N,2)

        # Distance→weights (t_groups,H,W,N)
        diff  = align_xy[:, None, None, :, :] - grid[None, :, :, None, :]    # (t_groups,H,W,N,2)
        dist  = (diff * diff).sum(-1)                                        # (t_groups,H,W,N)
        weight = torch.exp(-dist * temperature) * align_vis.squeeze(-1)[:, None, None, :]  # (t_groups,H,W,N)

        k = int(min(max(1, topk), weight.shape[-1]))
        vert_weight, vert_index = torch.topk(weight, k=k, dim=-1)            # (t_groups,H,W,k)

    # Sample point features from frame 0 latent (C=vae_divide[1]) at normalized coords of t=0
    x0_in = vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1]           # (1,C,H,W)
    g     = xy_n[:, :1].reshape(1, 1, N, 2).to(x0_in.dtype)       # (1,1,N,2)
    pt = F.grid_sample(x0_in, g, mode="bilinear", padding_mode="zeros", align_corners=False)  # (1,C,1,N)
    pt = pt.contiguous()
    point_feature = pt[0, :, 0, :].transpose(0, 1).contiguous()   # (N,C)

    # Merge & blend (keep original outputs for 81 frames)
    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2)  # (C,T-1,H,W)
    out_weight  = vert_weight.sum(-1)                                                      # (T-1,H,W)
    mix_feature = out_feature + vid[vae_divide[0]:, 1:] * (1 - out_weight.clamp(0, 1))
    out_feature_full = torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1)            # (C,T,H,W)
    out_mask_full    = torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0)     # (T,H,W)

    return torch.cat([out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full], dim=0)
