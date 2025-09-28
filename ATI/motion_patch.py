print(">>> USING motion_patch FROM:", __file__)

import torch
import torch.nn.functional as F

def merge_final(point_feature: torch.Tensor,
                vert_weight: torch.Tensor,
                vert_index: torch.Tensor) -> torch.Tensor:
    """
    point_feature: (N, C)            per-track features
    vert_weight:   (T-1, H, W, k)    weights for top-k tracks per pixel
    vert_index:    (T-1, H, W, k)    indices of top-k tracks per pixel
    returns:       (C, T-1, H, W)
    """
    # Sanity/dtype
    assert point_feature.dim() == 2, f"point_feature shape {tuple(point_feature.shape)} != (N,C)"
    assert vert_weight.shape == vert_index.shape, "vert_weight / vert_index shapes must match"
    assert vert_index.dtype in (torch.int32, torch.int64), "vert_index must be int"

    # Normalize weights across k to avoid over/under-scaling
    w = vert_weight
    wsum = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    w = w / wsum  # (T-1, H, W, k)

    # Gather per-track features for each top-k index -> (T-1, H, W, k, C)
    # Advanced indexing uses vert_index to select rows from (N, C)
    feats = point_feature[vert_index]  # (T-1, H, W, k, C)

    # Weighted sum over k -> (T-1, H, W, C)
    fused = (feats * w[..., None]).sum(dim=-2)

    # Reorder to (C, T-1, H, W)
    return fused.permute(3, 0, 1, 2).contiguous()

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
def patch_motion_tether(tracks, vid, topk=2, temperature=25.0, vae_divide=(16,)):
    """
    tracks: (B, T, N, 4)  [mask_or_dummy, x, y, visible]
    vid:    (C, T, H, W)
    return: (C + vae_divide[0], T, H, W)
    """
    import torch
    import torch.nn.functional as F

    def _to_strided(x):
        if hasattr(x, "layout") and x.layout != torch.strided:
            x = x.to_dense()
        return x.contiguous()

    def _resample_tracks_time(tr, T_out: int):
        B, T_in, N, D = tr.shape
        if T_in == T_out:
            return tr
        x = tr.permute(0, 2, 3, 1).reshape(B * N * D, 1, T_in).to(tr.dtype)
        x = torch.nn.functional.interpolate(x, size=T_out, mode="linear", align_corners=False)
        x = x.reshape(B, N, D, T_out).permute(0, 3, 1, 2).contiguous()
        return x

    vid    = _to_strided(vid)
    tracks = _to_strided(tracks)

    C, T, H, W = vid.shape
    if tracks.shape[1] != T:
        tracks = _resample_tracks_time(tracks, T)

    B, _, N, _ = tracks.shape

    # split xy / visible
    _, xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)        # (B,T,N,2), (B,T,N,1)
    s = float(min(H, W))
    norm = torch.tensor([W / s, H / s], device=vid.device, dtype=vid.dtype)
    xy_n = (xy / norm).clamp(-1, 1)
    visible = visible.clamp(0, 1)

    # grid for weights
    xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
    yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
    grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)
    grid = _to_strided(grid)

    # align to T-1
    xy_pad  = xy[:, 1:]        # (B,T-1,N,2)
    vis_pad = visible[:, 1:]   # (B,T-1,N,1)

    if B == 1:
        vis_sum = _to_strided(vis_pad.squeeze(0))           # (T-1,N,1)
        xy_sum  = _to_strided(xy_pad.squeeze(0) * vis_sum)  # (T-1,N,2)
    else:
        vis_sum = _to_strided(vis_pad.sum(0))               # (T-1,N,1)
        xy_sum  = _to_strided((xy_pad * vis_pad).sum(0))    # (T-1,N,2)

    eps = 1e-5
    align_vis = vis_sum                                     # (T-1,N,1)
    align_xy  = _to_strided(xy_sum / (align_vis + eps))     # (T-1,N,2)

    diff  = _to_strided(align_xy[:, None, None, :, :] - grid[None, :, :, None, :])  # (T-1,H,W,N,2)
    dist  = _to_strided((diff * diff).sum(-1))                                      # (T-1,H,W,N)
    vmask = _to_strided(align_vis.squeeze(-1))                                       # (T-1,N)
    weight = _to_strided(torch.exp(-dist * temperature) * vmask[:, None, None, :])  # (T-1,H,W,N)

    k = int(min(max(1, topk), weight.shape[-1]))
    vert_weight, vert_index = torch.topk(weight, k=k, dim=-1)                       # (T-1,H,W,k)
    vert_weight = _to_strided(vert_weight)
    vert_index  = _to_strided(vert_index)

    # robust point-feature extraction (no .t() / permute(1,0))
    x0 = _to_strided(vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1])   # (1,C,H,W)
    g  = _to_strided(xy_n[:, :1].reshape(1, 1, N, 2).to(x0.dtype))  # (1,1,N,2)

    pt = F.grid_sample(
        x0, g,
        mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (1,C,1,N)

    if getattr(pt, "is_sparse", False):
        pt = pt.to_dense()
    pt = pt.contiguous()
    pt = pt.view(pt.shape[0], pt.shape[1], -1)              # (1,C,N)
    point_feature = pt[0].transpose(0, 1).contiguous()      # (N,C)

    out_feature = _to_strided(merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2))  # (C,T-1,H,W)
    out_weight  = _to_strided(vert_weight.sum(-1))                                                          # (T-1,H,W)

    mix_feature = _to_strided(out_feature + vid[vae_divide[0]:, 1:] * (1 - out_weight.clamp(0, 1)))
    out_full    = _to_strided(torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1))                    # (C,T,H,W)
    mask_full   = _to_strided(torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0))             # (T,H,W)

    return torch.cat([mask_full[None].expand(vae_divide[0], -1, -1, -1), out_full], dim=0)


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

    def _to_strided(x):
        if hasattr(x, "layout") and x.layout != torch.strided:
            x = x.to_dense()
        return x.contiguous()

    def _resample_tracks_time(tr, T_out: int):
        # Linear time resample on the T dimension (keeps B,N,4 intact)
        B, T_in, N, D = tr.shape
        if T_in == T_out:
            return tr
        # (B, T, N, D) -> (B*N*D, 1, T)
        x = tr.permute(0, 2, 3, 1).reshape(B * N * D, 1, T_in).to(tr.dtype)
        x = torch.nn.functional.interpolate(x, size=T_out, mode="linear", align_corners=False)
        x = x.reshape(B, N, D, T_out).permute(0, 3, 1, 2).contiguous()
        return x

    with torch.no_grad():
        vid    = _to_strided(vid)
        tracks = _to_strided(tracks)

        C, T, H, W = vid.shape
        if tracks.shape[1] != T:
            tracks = _resample_tracks_time(tracks, T)

        B, _, N, _ = tracks.shape

        # Split xy + visibility
        _, xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)    # xy:(B,T,N,2), visible:(B,T,N,1)
        s = float(min(H, W))
        norm = torch.tensor([W / s, H / s], device=vid.device, dtype=vid.dtype)
        xy_n = (xy / norm).clamp(-1, 1)
        visible = visible.clamp(0, 1)

        # Build normalized grid (H,W,2) used for spatial weights
        xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
        yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)
        grid = _to_strided(grid)

        # Align to T-1
        xy_pad  = xy[:, 1:]        # (B,T-1,N,2)
        vis_pad = visible[:, 1:]   # (B,T-1,N,1)

        # Visibility / weighted xy across batch
        if B == 1:
            vis_sum = _to_strided(vis_pad.squeeze(0))            # (T-1,N,1)
            xy_sum  = _to_strided((xy_pad.squeeze(0) * vis_sum)) # (T-1,N,2)
        else:
            vis_sum = _to_strided(vis_pad.sum(0))                # (T-1,N,1)
            xy_sum  = _to_strided((xy_pad * vis_pad).sum(0))     # (T-1,N,2)

        eps = 1e-5
        align_vis = vis_sum                                      # (T-1,N,1)
        align_xy  = _to_strided(xy_sum / (align_vis + eps))      # (T-1,N,2)

        # Distance -> weights (T-1,H,W,N)
        diff  = _to_strided(align_xy[:, None, None, :, :] - grid[None, :, :, None, :])
        dist  = _to_strided((diff * diff).sum(-1))
        vmask = _to_strided(align_vis.squeeze(-1))               # (T-1,N)
        weight = _to_strided(torch.exp(-dist * temperature) * vmask[:, None, None, :])

        # Top-k over tracks
        k = int(min(max(1, topk), weight.shape[-1]))
        vert_weight, vert_index = torch.topk(weight, k=k, dim=-1)  # (T-1,H,W,k)
        vert_weight = _to_strided(vert_weight)
        vert_index  = _to_strided(vert_index)

    # === Point-feature extraction on frame 0 ===
    x0_in = _to_strided(vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1])       # (1,C,H,W)
    grid_ = _to_strided(xy_n[:, :1].reshape(1, 1, N, 2).to(x0_in.dtype))   # (1,1,NN,2)

    try:
        pt = F.grid_sample(
            x0_in, grid_,
            mode="bilinear", padding_mode="zeros", align_corners=False
        )  # -> (1, C, 1, N)
        pt = _to_strided(pt)
        # (1,C,1,N) -> (C,N) -> (N,C) without .t() / permute(1,0)
        point_feature = _to_strided(pt[0, :, 0, :]).transpose(0, 1)         # (N, C)
    except Exception:
        # Fallback: NN gather
        x0 = _to_strided(vid[vae_divide[0]:, 0])   # (C,H,W)
        xy0 = _to_strided(xy_n[:, 0, :, :])        # (B=1,N,2)
        if xy0.dim() == 3 and xy0.shape[0] == 1:
            xy0 = xy0[0]                           # (N,2)
        px = ((xy0[..., 0] + 1) * 0.5) * (W - 1)
        py = ((xy0[..., 1] + 1) * 0.5) * (H - 1)
        px = torch.clamp(px.round().long(), 0, W - 1)
        py = torch.clamp(py.round().long(), 0, H - 1)
        point_feature = _to_strided(x0[:, py, px].transpose(0, 1))          # (N,C)

    # Merge and blend
    out_feature = _to_strided(merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2))  # (C,T-1,H,W)
    out_weight  = _to_strided(vert_weight.sum(-1))                                                      # (T-1,H,W)

    mix_feature      = _to_strided(out_feature + vid[vae_divide[0]:, 1:] * (1 - out_weight.clamp(0, 1)))
    out_feature_full = _to_strided(torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1))            # (C,T,H,W)
    out_mask_full    = _to_strided(torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0))     # (T,H,W)

    return torch.cat(
        [out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full],
        dim=0
    )


