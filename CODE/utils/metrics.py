import torch

# -------- basic per-dimension errors on arbitrary tensors --------
def mae_per_dim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error per output dimension."""
    return (pred - target).abs().mean(dim=0)

def mse_per_dim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error per output dimension."""
    return ((pred - target) ** 2).mean(dim=0)

def rmse_per_dim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error per output dimension."""
    return mse_per_dim(pred, target).sqrt()

def denorm(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert z-score normalization."""
    return y * (std + 1e-8) + mean

# -------- helpers for 9D targets: [x,y,z, sy,cy, sp,cp, sr,cr] --------
def _split_pos_ang(y: torch.Tensor):
    """Split (B,9) into pos (B,3) and angle pairs (B,3,2)."""
    pos = y[:, :3]
    ang_pairs = y[:, 3:].view(-1, 3, 2)  # (sy,cy), (sp,cp), (sr,cr)
    return pos, ang_pairs

def _unit_pairs(pairs: torch.Tensor) -> torch.Tensor:
    """Normalize each (sin,cos) pair to unit length."""
    return pairs / (pairs.norm(dim=-1, keepdim=True) + 1e-8)

def angular_errors_deg(pred_pairs: torch.Tensor, targ_pairs: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample per-axis absolute angular error in degrees (B,3),
    from predicted/target (sin,cos) pairs.
    Uses atan2 of the relative rotation to avoid unwrap issues.
    """
    p = _unit_pairs(pred_pairs)
    t = _unit_pairs(targ_pairs)
    ps, pc = p[..., 0], p[..., 1]
    ts, tc = t[..., 0], t[..., 1]
    # signed delta angle in radians in [-pi, pi]
    dtheta = torch.atan2(ps * tc - pc * ts, pc * tc + ps * ts)
    return dtheta.abs() * (180.0 / torch.pi)  # (B,3)

def metrics_9d(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_mean: torch.Tensor | None = None,   # shape (3,) if you normalized positions
    pos_std: torch.Tensor | None = None,    # shape (3,)
    pos_unit_scale: float = 1.0             # e.g., 1000.0 to report in mm
) -> dict:
    """
    Returns a dict with position MAE/RMSE and angular MAE/RMSE in degrees.
    If pos_mean/std provided, de-normalizes the first 3 dims before computing errors.
    """
    pred_pos, pred_ang = _split_pos_ang(pred)
    targ_pos, targ_ang = _split_pos_ang(target)

    if (pos_mean is not None) and (pos_std is not None):
        pred_pos = denorm(pred_pos, pos_mean, pos_std)
        targ_pos = denorm(targ_pos, pos_mean, pos_std)

    # Position errors
    pos_err = pred_pos - targ_pos
    pos_mae = pos_err.abs().mean(dim=0) * pos_unit_scale      # (3,)
    pos_rmse = (pos_err.pow(2).mean(dim=0).sqrt()) * pos_unit_scale

    # Angular errors (degrees)
    ang_err_deg = angular_errors_deg(pred_ang, targ_ang)      # (B,3)
    ang_mae_deg = ang_err_deg.mean(dim=0)                     # (3,)
    ang_rmse_deg = ang_err_deg.pow(2).mean(dim=0).sqrt()      # (3,)

    return {
        "pos_mae": pos_mae,               # per-axis
        "pos_rmse": pos_rmse,             # per-axis
        "pos_mae_mean": pos_mae.mean(),   # scalar
        "pos_rmse_mean": pos_rmse.mean(), # scalar
        "ang_mae_deg": ang_mae_deg,       # [yaw,pitch,roll]
        "ang_rmse_deg": ang_rmse_deg,     # [yaw,pitch,roll]
        "ang_mae_deg_mean": ang_mae_deg.mean(),   # scalar
        "ang_rmse_deg_mean": ang_rmse_deg.mean(), # scalar
    }
