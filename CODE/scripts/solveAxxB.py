# solveAxxB — hand-eye from recorded runs (supports AprilTag or Chessboard NPZ logs)

import os, re, glob, time
import numpy as np
import cv2

# ---------- Config (single camera, SN-only) ----------
DATA_ROOT     = "../DATA"            # base DATA folder
CAMERA_SERIAL = "839112062097"       # your single camera SN

# ---------- Utils ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def list_run_dirs(base):
    """Find run#### folders (lowercase, no underscore)."""
    if not os.path.isdir(base): return []
    runs = [d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and re.match(r"^run\d{4}$", d)]
    # sort numerically
    return sorted(runs, key=lambda d: int(d.replace("run","")))

def rot_angle_deg(R):
    """Geodesic angle between R and I (degrees)."""
    v = (np.trace(R) - 1.0) / 2.0
    v = np.clip(v, -1.0, 1.0)
    return float(np.degrees(np.arccos(v)))

def summarize_base_to_tag(H_b_t_list):
    """Return metrics over a set of ^bT_t (or ^bT_board) estimates."""
    if len(H_b_t_list) == 0:
        return None
    T = np.stack([H[:3,3] for H in H_b_t_list], axis=0)  # (N,3)
    t_mean = T.mean(axis=0)
    t_std  = T.std(axis=0)
    t_rms  = float(np.sqrt(np.mean(np.sum((T - t_mean)**2, axis=1))))  # meters
    R0 = H_b_t_list[0][:3,:3]
    angs = [rot_angle_deg(R0.T @ H[:3,:3]) for H in H_b_t_list]
    ang_rms = float(np.sqrt(np.mean(np.square(angs))))
    ang_max = float(np.max(angs))
    return {"t_mean": t_mean, "t_std": t_std, "t_rms": t_rms,
            "ang_rms_deg": ang_rms, "ang_max_deg": ang_max, "N": len(H_b_t_list)}

def save_handeye_results(cam_root, serial, method_name, H_g_c, H_b_t_list, files_used_abs):
    """Append CSV + save NPZ under <cam_root>/handeye_results/"""
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(cam_root, "handeye_results")
    ensure_dir(out_dir)

    summary = summarize_base_to_tag(H_b_t_list) or {}
    t_mean = summary.get("t_mean", np.zeros(3))
    t_std  = summary.get("t_std",  np.zeros(3))
    t_rms  = summary.get("t_rms",  0.0)
    ang_rms = summary.get("ang_rms_deg", 0.0)
    ang_max = summary.get("ang_max_deg", 0.0)
    N = summary.get("N", 0)

    # CSV append
    csv_path = os.path.join(out_dir, "handeye_results.csv")
    new_file = not os.path.exists(csv_path)
    import csv as _csv
    with open(csv_path, "a", newline="") as f:
        w = _csv.writer(f)
        if new_file:
            w.writerow([
                "timestamp","serial","method","num_poses",
                "t_mean_x","t_mean_y","t_mean_z",
                "t_std_x","t_std_y","t_std_z",
                "t_rms_m","ang_rms_deg","ang_max_deg",
                *[f"Hgc_{r}{c}" for r in range(4) for c in range(4)],
                "files_used"
            ])
        w.writerow([
            ts, serial, method_name, N,
            *t_mean.tolist(), *t_std.tolist(),
            t_rms, ang_rms, ang_max,
            *H_g_c.reshape(-1).tolist(),
            ";".join([os.path.relpath(p, cam_root) for p in files_used_abs])
        ])

    # NPZ bundle
    npz_name = f"{ts}_handeye_{serial}_{method_name}.npz"
    np.savez_compressed(
        os.path.join(out_dir, npz_name),
        timestamp=ts,
        serial=serial,
        method=method_name,
        H_g_c=H_g_c,
        H_b_t=np.stack(H_b_t_list) if len(H_b_t_list) else np.zeros((0,4,4)),
        files_used=np.array([os.path.relpath(p, cam_root) for p in files_used_abs])
    )
    print(f"[LOG] Saved CSV: {os.path.abspath(csv_path)}")
    print(f"[LOG] Saved NPZ: {os.path.abspath(os.path.join(out_dir, npz_name))}")
    print(f"[LOG] Summary: N={N}  t_rms={t_rms*1000:.1f} mm  ang_rms={ang_rms:.3f}° (max={ang_max:.3f}°)")

# ---------- Locate runs for this serial ----------
cam_root = os.path.join(DATA_ROOT, f"CAM_{CAMERA_SERIAL}_calibration")
if not os.path.isdir(cam_root):
    raise SystemExit(f"No camera folder: {cam_root}")

runs = list_run_dirs(cam_root)
if not runs:
    raise SystemExit(f"No run folders under {cam_root}")

R_gripper2base, t_gripper2base = [], []
R_target2cam,   t_target2cam   = [], []
files_used = []

def _load_last_npz_in_run(run_dir, serial):
    """Return path to last NPZ for the given serial inside run_dir (timestamp sort)."""
    patterns = [
        os.path.join(run_dir, f"*_cam*_{serial}.npz"),
        os.path.join(run_dir, f"*_{serial}.npz"),
    ]
    matches = []
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            break
    return sorted(matches)[-1] if matches else None

for run in runs:
    run_dir = os.path.join(cam_root, run)
    fp = _load_last_npz_in_run(run_dir, CAMERA_SERIAL)
    if fp is None:
        print(f"WARNING: no NPZ for serial {CAMERA_SERIAL} in {run}")
        continue

    with np.load(fp, allow_pickle=True) as data:
        # Robot pose ^bT_g (aka ^bT_ee)
        if   "H_b_ee" in data.files: H_b_g = data["H_b_ee"]
        elif "H_robot" in data.files: H_b_g = data["H_robot"]
        else:
            print(f"SKIP: {os.path.basename(fp)} has no robot pose 'H_b_ee'/'H_robot'")
            continue

        # Camera->target pose: support both AprilTag and Chessboard logs
        H_c_t = None
        if "H_c_t" in data.files:
            # AprilTag path: may be a stack (N,4,4)
            H_c_t_stack = data["H_c_t"]
            if H_c_t_stack.ndim == 3 and H_c_t_stack.shape[-2:] == (4,4) and H_c_t_stack.shape[0] > 0:
                H_c_t = H_c_t_stack[0]
            elif H_c_t_stack.shape == (4,4):
                H_c_t = H_c_t_stack
        if H_c_t is None and "H_c_board" in data.files:
            # Chessboard path: single 4x4
            H_c_t = data["H_c_board"]

        if H_c_t is None:
            print(f"SKIP: {os.path.basename(fp)} has neither 'H_c_t' nor 'H_c_board'")
            continue

        # Normalize dtypes
        H_b_g = np.asarray(H_b_g, dtype=np.float64).reshape(4,4)
        H_c_t = np.asarray(H_c_t, dtype=np.float64).reshape(4,4)

        R_gripper2base.append(H_b_g[:3,:3])
        t_gripper2base.append(H_b_g[:3, 3].reshape(3,1))
        R_target2cam.append(H_c_t[:3,:3])
        t_target2cam.append(H_c_t[:3, 3].reshape(3,1))
        files_used.append(os.path.abspath(fp))

if len(R_gripper2base) < 3:
    raise SystemExit(f"Need several poses; got {len(R_gripper2base)}")

print(f"Using {len(files_used)} pose(s) from {len(runs)} run(s)")
for f in files_used:
    print(" -", f)

# ---------- Hand-eye solve ----------
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam,   t_target2cam,
    method=cv2.CALIB_HAND_EYE_PARK  # alternatives: DANIILIDIS, TSAI, HORAUD, ANDREFF
)

# ^gT_c (camera -> gripper)
H_g_c = np.eye(4, dtype=np.float64)
H_g_c[:3,:3] = R_cam2gripper
H_g_c[:3, 3] = t_cam2gripper.ravel()

print("\nEstimated ^gT_c (camera->gripper):")
print(np.array_str(H_g_c, precision=5, suppress_small=True))

# Build ^bT_t (or ^bT_board) for each used pose
H_b_t_list = []
for Rb, tb, Rt, tt in zip(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
    H_b_g = np.eye(4, dtype=np.float64); H_b_g[:3,:3] = Rb; H_b_g[:3,3] = tb.ravel()
    H_c_t = np.eye(4, dtype=np.float64); H_c_t[:3,:3] = Rt; H_c_t[:3,3] = tt.ravel()
    H_b_t = H_b_g @ H_g_c @ H_c_t
    H_b_t_list.append(H_b_t)

# Save results under this camera’s folder
method_name = "PARK"
save_handeye_results(cam_root, CAMERA_SERIAL, method_name,
                     H_g_c, H_b_t_list, files_used)

# If you want ^bT_c for the last pose:
H_b_g_last4 = np.eye(4, dtype=np.float64)
H_b_g_last4[:3,:3] = R_gripper2base[-1]
H_b_g_last4[:3,3]  = t_gripper2base[-1].ravel()
H_b_c = H_b_g_last4 @ H_g_c
print("\nDerived ^bT_c at last pose (base->camera):")
print(np.array_str(H_b_c, precision=5, suppress_small=True))
