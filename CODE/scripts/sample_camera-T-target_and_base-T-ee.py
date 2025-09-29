import argparse, sys, time, os, csv
import numpy as np
import cv2
import pyrealsense2 as rs
import re

# -----------------------------
# Config you asked for
# -----------------------------
CAMERA_SERIALS = ["839112062097"]  # known order

SAVE_PATH = "../../DATA/"
ROBOT_IP = "172.16.0.2"

# Default chessboard calibration params
BOARD_COLS = 8          # internal corners along X
BOARD_ROWS = 5          # internal corners along Y
SQUARE_SIZE_M = 0.02875    # 3 cm squares -> 0.03

# -----------------------------
# Utils
# -----------------------------
def aggregate_vis_dir(cam_root, name="ALL_CHESSBOARD_VIS"):
    p = os.path.join(cam_root, name)
    os.makedirs(p, exist_ok=True)
    return p

def next_run_dir(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and re.match(r"^run\d{4}$", d)
    ]
    if not existing:
        idx = 1
    else:
        nums = [int(d.replace("run", "")) for d in existing]
        idx = max(nums) + 1
    run_dir = os.path.join(base_dir, f"run{idx:04d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, idx


def rvec_tvec_to_transform(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = tvec.reshape(3)
    return T


def rot_to_euler_zyx(R):
    sy = -R[2,0]
    cy = np.sqrt(max(0.0, 1.0 - sy*sy))
    if cy > 1e-6:
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(sy, cy)
        roll  = np.arctan2(R[2,1], R[2,2])
    else:
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(sy, cy)
        roll  = 0.0
    return np.array([yaw, pitch, roll])


def draw_axes(img, K, D, rvec, tvec, axis_len=0.05):
    O = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]])
    pts, _ = cv2.projectPoints(O, rvec, tvec, K, D)
    pts = pts.reshape(-1,2).astype(int)
    cv2.line(img, pts[0], pts[1], (0, 0, 255), 2)   # X red
    cv2.line(img, pts[0], pts[2], (0, 255, 0), 2)   # Y green
    cv2.line(img, pts[0], pts[3], (255, 0, 0), 2)   # Z blue
    return img


def iso_ts():
    return time.strftime("%Y-%m-%dT%H-%M-%S")

# -----------------------------
# Chessboard pose estimation (PnP)
# -----------------------------

def estimate_chessboard_pose(
    image_bgr,
    K, D,
    board_cols, board_rows,                # inner corners (e.g., 8, 5 for 9x6 squares)
    square_size_m,                         # e.g., 0.03 for 3 cm
    try_swap_orientation=True,             # also try (rows, cols) if first fails
    use_sb=True
):
    """
    Detect a symmetric chessboard (inner corners = board_cols x board_rows),
    refine corners, estimate planar pose, and draw axes.

    Returns:
        (result, vis_bgr) where result is either a dict or None.
        If result is not None, it contains:
            - 'cols','rows','square_m'
            - 'corners'  (N,2) float32 image points
            - 'objp'     (N,3) float32 object points (Z=0 plane)
            - 'rvec','tvec' (3,), 'R' (3,3), 't' (3,)
            - 'H_c_board' (4,4) homogeneous transform
            - 'euler_zyx' (optional) if rot_to_euler_zyx available
    """
    import numpy as np
    import cv2

    # --- basic checks / normalize inputs ---
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        raise ValueError("estimate_chessboard_pose: empty image")

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D = None if D is None else np.asarray(D, dtype=np.float64).ravel()

    vis = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Mild histogram equalization can help under uneven lighting
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
    except Exception:
        gray_eq = gray

    # Try the given pattern; if allowed, also try swapped orientation
    patterns = [(int(board_cols), int(board_rows))]
    if try_swap_orientation and board_cols != board_rows:
        patterns.append((int(board_rows), int(board_cols)))

    # --- helpers for detection paths ---
    def _try_sb(g, pat):
        """findChessboardCornersSB with robust flags; supports various OpenCV return shapes."""
        if not hasattr(cv2, "findChessboardCornersSB"):
            return False, None
        out = cv2.findChessboardCornersSB(g, pat, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        # Newer OpenCV: (retval, corners)
        if isinstance(out, tuple):
            if len(out) == 2:
                retval, c = out
                return bool(retval), c
            # Older quirk: may return corners only
            c = out[-1]
            ok = (c is not None) and (len(c) == pat[0] * pat[1])
            return ok, c
        # Very old: corners only
        c = out
        ok = (c is not None) and (len(c) == pat[0] * pat[1])
        return ok, c

    def _try_classic(g, pat):
        """findChessboardCorners + cornerSubPix refinement."""
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, c = cv2.findChessboardCorners(g, pat, flags=flags)
        if not ok:
            return False, None
        # refine on the original (non-CLAHE) gray image
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        c_ref = cv2.cornerSubPix(gray, c, (11,11), (-1,-1), crit)
        return True, c_ref

    # --- detection loop ---
    found, corners, pattern_size = False, None, None
    for pat in patterns:
        if use_sb:
            found, corners = _try_sb(gray_eq, pat)
        if not found:
            found, corners = _try_classic(gray_eq, pat)
        if found and corners is not None and len(corners) == pat[0]*pat[1]:
            pattern_size = pat
            break

    if not found or corners is None or pattern_size is None:
        # nothing found—return the untouched frame so you can see what it saw
        return None, vis

    # Ensure shape (N,1,2) float32 for OpenCV
    if corners.dtype != np.float32:
        corners = corners.astype(np.float32)
    if corners.ndim == 2:
        corners = corners.reshape(-1, 1, 2)

    # Draw corners for visual confirmation
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)

    # --- build object points (Z=0 plane) matching OpenCV corner order ---
    cols, rows = pattern_size  # columns = along image width (x), rows = height (y)
    objp = np.zeros((rows * cols, 3), np.float32)
    # OpenCV orders corners row-major from top-left, matching this mgrid:
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp *= float(square_size_m)

    img_pts = corners.reshape(-1, 2)

    # --- pose estimation: favor IPPE (planar), fallback to SQPNP then ITERATIVE ---
    ok, rvec, tvec = False, None, None
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        ok, rvec, tvec = cv2.solvePnP(objp, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE)
    if not ok and hasattr(cv2, "SOLVEPNP_SQPNP"):
        ok, rvec, tvec = cv2.solvePnP(objp, img_pts, K, D, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(objp, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        # Return the visualization so you can inspect detection without crashing upstream
        return None, vis

    # --- construct H_c_board (4x4) ---
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    H_c_board = np.eye(4, dtype=float)
    H_c_board[:3, :3] = R
    H_c_board[:3,  3] = t

    # --- draw coordinate axes (scaled to board size) ---
    axis_len = max(0.05, 3*float(square_size_m))
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(vis, K, D, rvec, tvec, axis_len)
    else:
        # Fallback axis drawing
        O = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]])
        proj, _ = cv2.projectPoints(O, rvec, tvec, K, D)
        p = proj.reshape(-1,2).astype(int)
        cv2.line(vis, p[0], p[1], (0,0,255), 2)
        cv2.line(vis, p[0], p[2], (0,255,0), 2)
        cv2.line(vis, p[0], p[3], (255,0,0), 2)

    # --- annotate ---
    dist = float(np.linalg.norm(t))
    p0 = tuple(np.int32(corners[0,0]))
    label = f"{cols}x{rows} board  z={t[2]:.3f} m  d={dist:.3f} m"
    cv2.putText(vis, label, (max(5, p0[0]-100), max(20, p0[1]-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40,220,40), 2, cv2.LINE_AA)

    # Euler (optional): use your helper if present; otherwise compute here
    try:
        euler_zyx = rot_to_euler_zyx(R)  # uses your existing helper if defined above
    except NameError:
        # inline fallback (ZYX yaw-pitch-roll)
        sy = -R[2,0]
        cy = np.sqrt(max(0.0, 1.0 - sy*sy))
        if cy > 1e-6:
            yaw   = np.arctan2(R[1,0], R[0,0])
            pitch = np.arctan2(sy, cy)
            roll  = np.arctan2(R[2,1], R[2,2])
        else:
            yaw   = np.arctan2(-R[0,1], R[1,1])
            pitch = np.arctan2(sy, cy)
            roll  = 0.0
        euler_zyx = np.array([yaw, pitch, roll], dtype=float)

    result = {
        "cols": int(cols),
        "rows": int(rows),
        "square_m": float(square_size_m),
        "corners": corners.reshape(-1, 2),   # (N,2)
        "objp": objp,                        # (N,3)
        "rvec": rvec.reshape(3),
        "tvec": tvec.reshape(3),
        "R": R,
        "t": t,
        "H_c_board": H_c_board,
        "euler_zyx": euler_zyx,
    }
    return result, vis







# -----------------------------
# RealSense capture & intrinsics
# -----------------------------
def build_config(serial: str, w: int, h: int, fps: int):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    return pipe, cfg


def get_one_color_frame(pipe: rs.pipeline, warmup: int = 10, timeout_s: float = 5.0):
    for _ in range(max(0, warmup)):  # let auto-exposure settle
        pipe.wait_for_frames()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frames = pipe.wait_for_frames()
        c = frames.get_color_frame()
        if c:
            return c
    return None


def intrinsics_from_profile(profile: rs.pipeline_profile):
    vsp = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intr = vsp.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0,       0,       1]], dtype=np.float32)
    D = np.array(intr.coeffs[:5], dtype=np.float32)  # (k1,k2,p1,p2,k3)
    return K, D, intr.width, intr.height, intr.model, np.array(intr.coeffs, dtype=np.float32)

# -----------------------------
# Optional robot pose
# -----------------------------
def get_robot_state(robot_ip: str | None):
    if not robot_ip:
        return None
    try:
        import time as _time
        import panda_py
        from panda_py import Panda
        robot = Panda(hostname=robot_ip)

        H = None
        try:
            H = np.array(robot.get_pose(), dtype=float)
        except Exception:
            pose = robot.pose()
            if isinstance(pose, dict) and "T" in pose:
                H = np.array(pose["T"], dtype=float)
            else:
                H = np.array(pose, dtype=float)

        q = None
        try:
            q = np.array(robot.get_joint_positions(), dtype=float)
        except Exception:
            try:
                q = np.array(robot.joint_positions(), dtype=float)
            except Exception:
                pass

        return {
            "frame": "base->ee",
            "H": H,
            "q": q,
            "timestamp": _time.strftime("%Y-%m-%dT%H-%M-%S"),
        }
    except Exception as e:
        print(f"[Robot] Failed to get pose: {e}", file=sys.stderr)
        return None

# -----------------------------
# Logging helpers
# -----------------------------
def ensure_dir(path): os.makedirs(path, exist_ok=True)

def append_csv(csv_path, rows, header):
    new_file = not os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path) or ".")
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def save_npz(npz_path, **kw):
    ensure_dir(os.path.dirname(npz_path) or ".")
    np.savez_compressed(npz_path, **kw)

# -----------------------------
# Per-camera capture + log
# -----------------------------
def capture_from_camera(
    serial, idx, w, h, fps, warmup, preview, save_dir,
    board_cols, board_rows, square_size_m,
    csv_log, npz_log_dir, robot_ip, aggregate_vis_dir=None

):
    ensure_dir(save_dir)
    ts = iso_ts()

    pipe, cfg = build_config(serial, w, h, fps)
    try:
        profile = pipe.start(cfg)
        frame = get_one_color_frame(pipe, warmup=warmup, timeout_s=5.0)
        if frame is None:
            print(f"[Cam {idx}] Timed out waiting for frame.", file=sys.stderr)
            return

        img = np.asanyarray(frame.get_data())
        K, D, ww, hh, model, coeffs_full = intrinsics_from_profile(profile)

        # Detect + pose (chessboard)
        result, vis = estimate_chessboard_pose(
            img, K, D,
            board_cols=board_cols,
            board_rows=board_rows,
            square_size_m=square_size_m
        )

        # Save images
        raw_name = f"{ts}_raw_cam{idx}_{serial}_{ww}x{hh}.png"
        vis_name = f"{ts}_chessboard_cam{idx}_{serial}_{ww}x{hh}.png"
        raw_path = os.path.join(save_dir, raw_name)
        vis_path = os.path.join(save_dir, vis_name)
        cv2.imwrite(raw_path, img)
        cv2.imwrite(vis_path, vis)
        print(f"[Cam {idx}] Saved raw: {os.path.abspath(raw_path)}")
        print(f"[Cam {idx}] Saved vis: {os.path.abspath(vis_path)}")

        if aggregate_vis_dir:
            # Optional: include run folder name in the filename for quick traceability
            run_folder_name = os.path.basename(os.path.normpath(save_dir))  # e.g., "run0021"
            agg_name = f"{run_folder_name}_{vis_name}"
            agg_path = os.path.join(aggregate_vis_dir, agg_name)
            cv2.imwrite(agg_path, vis)
            print(f"[Cam {idx}] Saved vis (aggregate): {os.path.abspath(agg_path)}")
        # Optional robot state (once per capture)
        robot_state = get_robot_state(robot_ip)

        # ---- NPZ (rich log, one file per camera-shot) ----
        if npz_log_dir:
            npz_name = f"{ts}_cam{idx}_{serial}.npz"
            npz_path = os.path.join(npz_log_dir, npz_name)

            payload = dict(
                timestamp=ts,
                cam_index=idx,
                serial=serial,
                image_size=np.array([hh, ww], int),
                K=K, D=D,
                rs_model=model,
                rs_coeffs=coeffs_full,
                board_cols=int(board_cols),
                board_rows=int(board_rows),
                square_size_m=float(square_size_m),
                raw_image=os.path.abspath(raw_path),
                vis_image=os.path.abspath(vis_path),
                robot_present=robot_state is not None,
            )

            if result is not None:
                payload.update(
                    corners=result["corners"],
                    objp=result["objp"],
                    H_c_board=result["H_c_board"],
                    R=result["R"],
                    t=result["t"],
                    rvec=result["rvec"],
                    tvec=result["tvec"],
                    euler_zyx=result["euler_zyx"]
                )

            if robot_state is not None:
                payload["robot_frame"] = robot_state["frame"]
                payload["H_b_ee"] = robot_state["H"]
                if robot_state.get("q") is not None:
                    payload["robot_q"] = robot_state["q"]
                if robot_state.get("timestamp") is not None:
                    payload["robot_timestamp"] = robot_state["timestamp"]

            save_npz(npz_path, **payload)
            print(f"[Cam {idx}] Saved NPZ log: {os.path.abspath(npz_path)}")

        # ---- CSV (one row per capture) ----
        if csv_log:
            header = [
                "timestamp","cam_index","serial","width","height",
                "K00","K01","K02","K10","K11","K12","K20","K21","K22",
                "Dk1","Dk2","Dp1","Dp2","Dk3",
                "board_cols","board_rows","square_m",
                *[f"H{r}{c}" for r in range(4) for c in range(4)],
                "tx","ty","tz","yaw","pitch","roll",
                "raw_path","vis_path",
                "robot_frame","robot_present",
                *[f"Hrobot{r}{c}" for r in range(4) for c in range(4)]
            ]
            rows = []

            if result is None:
                H_empty = np.zeros((4,4))
                row = [
                    ts, idx, serial, ww, hh,
                    *K.flatten().tolist(),
                    *D.tolist(),
                    int(board_cols), int(board_rows), float(square_size_m),
                    *H_empty.flatten().tolist(),
                    0,0,0, 0,0,0,
                    os.path.abspath(raw_path), os.path.abspath(vis_path),
                    (robot_state["frame"] if robot_state else ""), int(robot_state is not None),
                    *(robot_state["H"].flatten().tolist() if robot_state else [0]*16)
                ]
                rows.append(row)
            else:
                row = [
                    ts, idx, serial, ww, hh,
                    *K.flatten().tolist(),
                    *D.tolist(),
                    int(board_cols), int(board_rows), float(square_size_m),
                    *result["H_c_board"].flatten().tolist(),
                    *result["t"].tolist(),
                    *result["euler_zyx"].tolist(),
                    os.path.abspath(raw_path), os.path.abspath(vis_path),
                    (robot_state["frame"] if robot_state else ""), int(robot_state is not None),
                    *(robot_state["H"].flatten().tolist() if robot_state else [0]*16)
                ]
                rows.append(row)

            append_csv(csv_log, rows, header)
            print(f"[Cam {idx}] Appended {len(rows)} row(s) to CSV: {os.path.abspath(csv_log)}")

        # Console printout
        if result is None:
            print(f"[Cam {idx}] Chessboard not found.")
        else:
            print(f"\n[Cam {idx}] Chessboard {board_cols}x{board_rows} detected.")
            print("^cT_board (homogeneous):")
            print(np.array_str(result["H_c_board"], precision=4, suppress_small=True))
            print("t (m):", np.array_str(result["t"], precision=4))
            print("Euler ZYX (rad):", np.array_str(result["euler_zyx"], precision=4))

        if preview:
            cv2.imshow(f"Cam {idx} ({serial}) raw", img)
            cv2.imshow(f"Cam {idx} ({serial}) chessboard", vis)
            print("Press any key to close…")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    finally:
        try: pipe.stop()
        except Exception: pass

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Capture one frame, detect chessboard, save photos + NPZ into a per-camera run folder.")
    # Choose the camera
    p.add_argument("--serial", "-s", default=None,
                   help="Exact RealSense serial to use. If omitted, uses --cam-index or the first in CAMERA_SERIALS.")
    p.add_argument("--cam-index", "-i", type=int, default=None,
                   help=f"Camera index in CAMERA_SERIALS (0..{len(CAMERA_SERIALS)-1}).")

    # Image / runtime options
    p.add_argument("--res", default="1280x720", help="Resolution WxH (e.g., 640x480, 1280x720)")
    p.add_argument("--fps", type=int, default=30, help="Frame rate")
    p.add_argument("--warmup", type=int, default=12, help="Warm-up frames for auto-exposure")
    p.add_argument("--preview", action="store_true", help="Show windows")

    # Chessboard parameters
    p.add_argument("--board-cols", type=int, default=BOARD_COLS, help="Internal corners along width (columns). Default 9.")
    p.add_argument("--board-rows", type=int, default=BOARD_ROWS, help="Internal corners along height (rows). Default 6.")
    p.add_argument("--square-size", type=float, default=SQUARE_SIZE_M, help="Square size in meters. Default 0.03 (3 cm).")

    # Optional CSV
    p.add_argument("--csv-log", default=None,
                   help="If set, append a CSV here (e.g., '../DATA/CAM_<sn>_calibration/calib_log.csv').")

    # Data root (DATA/)
    p.add_argument("--data-root", default=SAVE_PATH,
                   help="Base DATA/ directory. Script will create DATA/CAM_<serial>_calibration/run#### folders under it.")

    # Robot (optional)
    p.add_argument("--robot-ip", default=None,
                   help="If provided, attempts to read and log robot pose (e.g., 172.16.0.2)")

    args = p.parse_args()

    # Parse res
    try:
        w, h = map(int, args.res.lower().split("x"))
    except Exception:
        print(f"Invalid --res '{args.res}'. Use WxH, e.g., 1280x720.", file=sys.stderr)
        sys.exit(2)

    # Resolve exactly one camera
    if args.serial is not None:
        if args.serial not in CAMERA_SERIALS:
            print(f"Serial {args.serial} is not in CAMERA_SERIALS: {CAMERA_SERIALS}", file=sys.stderr)
            sys.exit(1)
        serial = args.serial
        idx = CAMERA_SERIALS.index(serial)
    elif args.cam_index is not None:
        if not (0 <= args.cam_index < len(CAMERA_SERIALS)):
            print(f"Invalid cam-index {args.cam_index}, must be 0..{len(CAMERA_SERIALS)-1}", file=sys.stderr)
            sys.exit(1)
        idx = args.cam_index
        serial = CAMERA_SERIALS[idx]
    else:
        # default to first known camera
        idx, serial = 0, CAMERA_SERIALS[0]

    # Build per-camera base folder: DATA/CAM_<serial>_calibration
    cam_root = os.path.join(args.data_root, f"CAM_{serial}_calibration")
    os.makedirs(cam_root, exist_ok=True)

    # Create the next run folder: .../run0001
    run_dir, run_idx = next_run_dir(cam_root)
    print(f"[OUT] Using run folder: {os.path.abspath(run_dir)}")

    # Create the aggregate annotated-photos folder (sibling to run####)
    agg_vis_dir = aggregate_vis_dir(cam_root, "ALL_CHESSBOARD_VIS")
    print(f"[OUT] Aggregate annotated folder: {os.path.abspath(agg_vis_dir)}")


    # CSV: if requested but is a directory, make a file inside it
    csv_path = None
    if args.csv_log:
        if os.path.isdir(args.csv_log):
            csv_path = os.path.join(args.csv_log, "calib_log.csv")
        else:
            os.makedirs(os.path.dirname(args.csv_log) or ".", exist_ok=True)
            csv_path = args.csv_log

    robot_ip = args.robot_ip if args.robot_ip else ROBOT_IP

    # Capture ONLY this camera, saving photos + NPZ into the same run folder
    capture_from_camera(
        serial, idx, w, h, args.fps, args.warmup,
        args.preview,
        save_dir=run_dir,             # photos go here
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        square_size_m=args.square_size,
        csv_log=csv_path,             # None → no CSV, else append here
        npz_log_dir=run_dir,          # NPZ goes here too
        robot_ip=robot_ip,
        aggregate_vis_dir=agg_vis_dir   # <-- NEW

    )

if __name__ == "__main__":
    main()
