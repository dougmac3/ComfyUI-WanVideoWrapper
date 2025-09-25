print(">>> USING motion_patch FROM:", __file__)

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
    tracks: (B, T, N, 4)  last dim = [mask_or_dummy, x, y, visible]
    vid:    (C, T, H, W)
    return: (C + vae_divide[0], T, H, W)
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        C, T, H, W = vid.shape
        B = tracks.shape[0]
        N = tracks.shape[2]

        # Split out xy + visibility
        _, xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)  # xy:(B,T,N,2), visible:(B,T,N,1)
        s = float(min(H, W))
        norm = torch.tensor([W / s, H / s], device=vid.device, dtype=vid.dtype)
        xy_n = (xy / norm).clamp(-1, 1)
        visible = visible.clamp(0, 1)

        # Build normalized grid (H,W,2) for distance weighting
        xx = torch.linspace(-W / s, W / s, W, device=vid.device, dtype=vid.dtype)
        yy = torch.linspace(-H / s, H / s, H, device=vid.device, dtype=vid.dtype)
        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1)  # (H,W,2)

        # Align to T-1
        xy_pad = xy[:, 1:]       # (B,T-1,N,2)
        vis_pad = visible[:, 1:] # (B,T-1,N,1)

        # Visibility and weighted xy across batch
        if B == 1:
            vis_sum = vis_pad.squeeze(0)                 # (T-1,N,1)
            xy_sum  = (xy_pad.squeeze(0) * vis_sum)      # (T-1,N,2)
        else:
            vis_sum = vis_pad.sum(0)                     # (T-1,N,1)
            xy_sum  = (xy_pad * vis_pad).sum(0)          # (T-1,N,2)

        eps = 1e-5
        align_vis = vis_sum                                # (T-1,N,1)
        align_xy  = xy_sum / (align_vis + eps)             # (T-1,N,2)

        # Distance to each track point -> weights
        # (T-1,N,2) vs (H,W,2) -> (T-1,H,W,N)
        diff  = align_xy[:, None, None, :, :] - grid[None, :, :, None, :]
        dist  = (diff * diff).sum(-1)                      # (T-1,H,W,N)
        vmask = align_vis.squeeze(-1)                      # (T-1,N)
        weight = torch.exp(-dist * temperature) * vmask[:, None, None, :]  # (T-1,H,W,N)

        # Top-k over tracks
        k = int(min(max(1, topk), weight.shape[-1]))
        vert_weight, vert_index = torch.topk(weight, k=k, dim=-1)  # (T-1,H,W,k)

    # --- Robust point-feature extraction ---
    # Input to grid_sample must be (N_in, C, H, W) and grid (N_in, H_out, W_out, 2).
    # We sample from the first time slice features (keeps behavior consistent with your pipeline).
    x_in   = vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1]   # (1, C, H, W)
    g_in   = xy_n[:, :1]                                   # (B, 1, N, 2)
    # Make grid shape (1, 1, N, 2) for N_in=1
    grid_  = g_in.reshape(1, 1, N, 2)                      # (1, 1, N, 2)

    pt = F.grid_sample(
        x_in, grid_.type(x_in.dtype),
        mode="bilinear", padding_mode="zeros", align_corners=False
    )  # -> (1, C, 1, N)

    # Make absolutely sure it's dense/contiguous and shaped to (N, C)
    if pt.is_sparse:
        pt = pt.to_dense()
    pt = pt.contiguous()
    if pt.dim() != 4:
        print(f"[ATI] Unexpected grid_sample rank: {pt.dim()} shape={tuple(pt.shape)}")
    # (1,C,1,N) -> (C,1,N)
    pt = pt.squeeze(0).contiguous()
    # (C,1,N) -> (C,N)
    pt = pt.reshape(pt.shape[0], -1).contiguous()
    # (C,N) -> (N,C)
    point_feature = pt.transpose(0, 1).contiguous()        # (N, C)

    # Merge per-pixel using provided helper
    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2)  # (C,T-1,H,W)
    out_weight  = vert_weight.sum(-1)                                                         # (T-1,H,W)

    # Blend with original features and reattach first frame
    mix_feature     = out_feature + vid[vae_divide[0]:, 1:] * (1 - out_weight.clamp(0, 1))
    out_feature_full = torch.cat([vid[vae_divide[0]:, :1], mix_feature], dim=1)              # (C,T,H,W)
    out_mask_full    = torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0)       # (T,H,W)

    # Prepend mask channels for VAE split
    return torch.cat(
        [out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full],
        dim=0
    )
