import torch
import torch.nn.functional as F

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
    Robust ATI motion patch that tolerates sampler/ATI step mismatches.
    tracks: (B, T, N, 4) last dim = [mask_or_dummy, x, y, visible] (mask is ignored)
    vid   : (C, T, H, W)
    returns: (C + vae_divide[0], T, H, W)  # mask channels prepended (unchanged contract)
    """
    C, T, H, W = vid.shape
    B = tracks.shape[0]
    N = tracks.shape[2]

    # Split to coords/visibility; ignore the first (mask/dummy)
    _, xy, vis = torch.split(tracks, [1, 2, 1], dim=-1)              # xy:(B,T,N,2) vis:(B,T,N,1)

    # Normalize coords to [-1,1] using min(H,W) scale (same convention as before)
    s = float(min(H, W))
    norm = torch.tensor([W / s, H / s], device=vid.device, dtype=vid.dtype)
    xy_n = (xy.to(vid.dtype) / norm).clamp(-1, 1)                     # (B,T,N,2)
    vis = vis.clamp(0, 1)                                             # (B,T,N,1)

    # Build grid (H,W,2) once
    xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
    yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
    grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)

    # Align to (T-1): drop first frame
    xy_pad = xy[:, 1:]                                                # (B, T-1, N, 2)
    vis_pad = vis[:, 1:]                                              # (B, T-1, N, 1)
    Tm1 = int(vis_pad.shape[1])                                       # robust T-1

    # Visibility-weighted average position per track, reduced over batch if B>1
    if B == 1:
        vis_sum = vis_pad.squeeze(0)                                  # (T-1, N, 1)
        xy_sum  = (xy_pad.squeeze(0) * vis_sum)                       # (T-1, N, 2)
    else:
        vis_sum = vis_pad.sum(0)                                      # (T-1, N, 1)
        xy_sum  = (xy_pad * vis_pad).sum(0)                           # (T-1, N, 2)

    eps = 1e-5
    align_vis = vis_sum                                               # (T-1, N, 1)
    align_xy  = xy_sum / (align_vis + eps)                            # (T-1, N, 2)

    # Distance of each pixel to each track point -> weights
    # align_xy: (T-1,N,2), grid: (H,W,2) -> (T-1,H,W,N,2) -> sum2 -> (T-1,H,W,N)
    diff = align_xy[:, None, None, :, :] - grid[None, :, :, None, :]
    dist = (diff * diff).sum(-1)                                      # (T-1, H, W, N)

    # Only consider visible tracks at that time
    vis_mask = align_vis.squeeze(-1).clamp(0, 1)                      # (T-1, N)
    weight = torch.exp(-dist * float(temperature)) * vis_mask[:, None, None, :]  # (T-1,H,W,N)

    # Top-k per pixel over N
    K = int(min(max(1, int(topk)), weight.shape[-1]))
    vert_weight, vert_index = torch.topk(weight, k=K, dim=-1)         # (T-1,H,W,K)

    # Sample a per-track feature at FIRST frame (matches existing behavior/contract)
    # vid: (C,T,H,W) -> take latent channels after VAE split
    latent = vid[vae_divide[0]:]                                      # (C,T,H,W)

    # grid_sample expects input (N,C,H,W) and grid (N,Ho,Wo,2); we want N=1
    # Use the first time-step’s normalized coords to sample per-track features
    # xy_n: (B,T,N,2) -> take [:, :1, ...] and cast to proper dtype
    pt = F.grid_sample(
        latent.permute(1, 0, 2, 3)[:1],                               # (1,C,H,W)
        xy_n[:, :1].to(vid.dtype),                                    # (B,1,N,2)
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    # -> (1,C,1,N) -> (N,C)
    point_feature = pt.squeeze(0).squeeze(0).permute(1, 0).contiguous()

    # Fuse: (C,T-1,H,W) without any 4-group reshape
    out_feature = _weighted_gather_fuse(point_feature, vert_weight, vert_index)  # (C,T-1,H,W)
    out_weight  = vert_weight.sum(-1).clamp(0, 1)                                 # (T-1,H,W)

    # Soft blend with original latent features for T-1 frames, then reattach frame 0
    mix_feature     = out_feature + latent[:, 1:] * (1.0 - out_weight)           # (C,T-1,H,W)
    out_feature_all = torch.cat([latent[:, :1], mix_feature], dim=1)             # (C,T,H,W)

    # Build mask stack expected upstream: (vae_divide[0], T, H, W) + features
    mask_full = torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0)   # (T,H,W)
    mask_full = mask_full[None].expand(vae_divide[0], -1, -1, -1).contiguous()    # (M,T,H,W)

    return torch.cat([mask_full, out_feature_all], dim=0)
