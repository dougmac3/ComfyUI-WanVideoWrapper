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
def patch_motion(tracks, vid, topk=2, temperature=25.0, vae_divide=(16,)):
    """
    tracks: (B, T, N, 4)  last dim = [mask_or_dummy, x, y, visible]
    vid:    (C, T, H, W)
    return: (C + vae_divide[0], T, H, W)
    """
    import torch
    import torch.nn.functional as F

    # --- local helper: resample tracks along time to match video T ---
    def _resample_tracks_time(tracks_in, target_T):
        # tracks_in: (B, T, N, 4)
        if not isinstance(tracks_in, torch.Tensor):
            tracks_in = torch.tensor(tracks_in)
        assert tracks_in.dim() == 4 and tracks_in.size(-1) == 4, f"expected (B,T,N,4), got {tuple(tracks_in.shape)}"

        B, T_in, N, C = tracks_in.shape  # C==4
        x = tracks_in.permute(0, 2, 3, 1).contiguous().reshape(B * N, C, T_in).to(torch.float32)  # (B*N,4,T)
        x = F.interpolate(x, size=target_T, mode="linear", align_corners=False)                    # (B*N,4,target_T)
        out = x.reshape(B, N, C, target_T).permute(0, 3, 1, 2).contiguous()                        # (B,target_T,N,4)
        return out.to(tracks_in.dtype)

    @torch.no_grad()
    def _to_strided(x):
        if hasattr(x, "layout") and x.layout != torch.strided:
            x = x.to_dense()
        return x.contiguous()

    with torch.no_grad():
        # Sanitize and align time lengths first
        vid    = _to_strided(vid)
        tracks = _to_strided(tracks)

        C, T, H, W = vid.shape
        if tracks.shape[1] != T:
            tracks = _resample_tracks_time(tracks, T)

        B, _, N, _ = tracks.shape

        # Split xy + visibility
        _, xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)   # xy:(B,T,N,2)  visible:(B,T,N,1)
        s = float(min(H, W))
        norm = torch.tensor([W / s, H / s], device=vid.device, dtype=vid.dtype)
        xy_n = (xy / norm).clamp(-1, 1)
        visible = visible.clamp(0, 1)

        # Grid over image for distance weighting
        xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
        yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)
        grid = _to_strided(grid)

        # Align to T-1
        xy_pad = xy[:, 1:]        # (B,T-1,N,2)
        vis_pad = visible[:, 1:]  # (B,T-1,N,1)

        # Accumulate across batch
        if B == 1:
            vis_sum = _to_strided(vis_pad.squeeze(0))            # (T-1,N,1)
            xy_sum  = _to_strided(xy_pad.squeeze(0) * vis_sum)   # (T-1,N,2)
        else:
            vis_sum = _to_strided(vis_pad.sum(0))                # (T-1,N,1)
            xy_sum  = _to_strided((xy_pad * vis_pad).sum(0))     # (T-1,N,2)

        eps = 1e-5
        align_vis = vis_sum                                      # (T-1,N,1)
        align_xy  = _to_strided(xy_sum / (align_vis + eps))      # (T-1,N,2)

        # Distance -> weights (T-1,H,W,N)
        diff   = _to_strided(align_xy[:, None, None, :, :] - grid[None, :, :, None, :])
        dist   = _to_strided((diff * diff).sum(-1))
        vmask  = _to_strided(align_vis.squeeze(-1))              # (T-1,N)
        weight = _to_strided(torch.exp(-dist * temperature) * vmask[:, None, None, :])

        # Top-k over tracks
        k = int(min(max(1, topk), weight.shape[-1]))
        vert_weight, vert_index = torch.topk(weight, k=k, dim=-1)    # (T-1,H,W,k)
        vert_weight = _to_strided(vert_weight)
        vert_index  = _to_strided(vert_index)

    # === point-feature extraction on first frame ===
    x0_in = _to_strided(vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1])      # (1,C,H,W)
    grid_ = _to_strided(xy_n[:, :1].reshape(1, 1, N, 2).to(x0_in.dtype))  # (1,1,N,2)

    try:
        pt = F.grid_sample(
            x0_in, grid_,
            mode="bilinear", padding_mode="zeros", align_corners=False
        )  # -> (1, C, 1, N)
        pt = _to_strided(pt)
        point_feature = _to_strided(pt[0, :, 0, :]).transpose(0, 1)  # (N, C)
    except Exception:
        # fallback: nearest-neighbor gather
        x0 = _to_strided(vid[vae_divide[0]:, 0])   # (C,H,W)
        xy0 = _to_strided(xy_n[:, 0, :, :])
        if xy0.dim() == 3 and xy0.shape[0] == 1:
            xy0 = xy0[0]                           # (N,2)
        px = ((xy0[..., 0] + 1) * 0.5) * (W - 1)
        py = ((xy0[..., 1] + 1) * 0.5) * (H - 1)
        px = torch.clamp(px.round().long(), 0, W - 1)
        py = torch.clamp(py.round().long(), 0, H - 1)
        point_feature = _to_strided(x0[:, py, px].transpose(0, 1))  # (N, C)

    # Merge per-pixel (merge_final is expected to be defined in this module)
    out_feature = _to_strided(merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2))  # (C,T-1,H,W)
    out_weight  = _to_strided(vert_weight.sum(-1))                                                      # (T-1,H,W)

    # Blend & reattach first frame — time dims now agree: both T-1
    mix_feature      = _to_strided(out_feature + vid[vae_divide[0]:, 1:] * (1 - out_weight.clamp(0, 1)))
    out_feature_full = _to_strided(torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1))  # (C,T,H,W)
    out_mask_full    = _to_strided(torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0))  # (T,H,W)

    # Prepend mask channels for VAE split
    return torch.cat(
        [out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full],
        dim=0
    )
