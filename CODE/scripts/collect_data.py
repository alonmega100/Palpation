#!/usr/bin/env python3
import os, sys, time, json, math, datetime
import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector

# === Optional: if you already have helpers like to_H/inv_H, remove these ===
def to_H(R, t):
    H = np.eye(4, dtype=float)
    H[:3,:3] = R
    H[:3, 3] = t.reshape(3)
    return H

# === CONFIG ===
ROBOT_IP = "172.16.0.2"
SERIAL = "839112062097"
FRAME_W, FRAME_H, FPS = 1280, 720, 30

# AprilTag config
OBJ_TAG_SIZE = 0.023  # meters (outer black border)
VALID_TAG_IDS = {0, 1, 2, 3}  # which IDs count as "seen"

# Camera intrinsics file (if available)
DATA_ROOT = "../DATA"
INTRINSICS_FILE = os.path.join(DATA_ROOT, f"Camera_intrinsics_{SERIAL}_Res{FRAME_W}x{FRAME_H}.npz")


ORI_NOISE_STD_DEG = 5.0      # std dev for yaw/pitch/roll noise (deg)
ORI_NOISE_CLAMP_DEG = 180.0  # keep noisy angles in [-clamp, +clamp]
RNG_SEED = 1234


# Robot motion & grid
APPROACH_PAUSE_S   = 0.25
MAX_DETECT_FRAMES  = 10
DETECT_MIN_AREA_PX = 50

# XY + Z grid (centered around current pose)
GRID_NX, GRID_NY, GRID_NZ = 5, 5, 3             # counts along X, Y, Z
GRID_DX, GRID_DY, GRID_DZ = 0.1, 0.1, 0.15    # spacing in meters
GRID_ORDER_XY = "snake"                         # "raster" or "snake"
Z_TOP_TO_BOTTOM = True                          # visit top Z first, then down


# Euler order Z-Y-X (yaw, pitch, roll)
YAW_LIST_DEG   = [  0,  25, -25 ]   # about base Z
PITCH_LIST_DEG = [  0,  25, -25 ]   # about base Y
ROLL_LIST_DEG  = [  0, 25, -25 ]


# Build orientation combinations (Z-Y-X)
ORI_LISTS = [(y, p, r) for y in YAW_LIST_DEG for p in PITCH_LIST_DEG for r in ROLL_LIST_DEG]

_rng = np.random.default_rng(RNG_SEED)

KEEP_Z_ORIENT_FROM_START = True                 # keep start Z and orientation
SAVE_OVERLAY = True
RUN_DIR = os.path.join(DATA_ROOT, "grid_capture_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

# === ROBOT (panda_py) ===
try:
    import panda_py
    robot = panda_py.Panda(hostname=ROBOT_IP)
    print(f"[robot] Connected to Panda at {ROBOT_IP}")
except Exception as e:
    print(f"[robot] ERROR connecting to Panda: {e}")
    sys.exit(1)


def _clamp_angle_deg(a, limit=180.0):
    if limit is None:
        return a
    return float(np.clip(a, -abs(limit), abs(limit)))

def get_pose_b_ee():
    H = np.array(robot.get_pose(), dtype=float)
    if H.shape != (4,4): raise RuntimeError("robot.get_pose() did not return 4x4")
    return H

def move_to_pose_b_ee(H_b_ee, speed=0.2):
    # Accepts a 4x4 pose and uses the correct keyword name: speed_factor
    H_b_ee = np.asarray(H_b_ee, dtype=float)
    assert H_b_ee.shape == (4,4), "move_to_pose_b_ee expects a 4x4 pose"
    robot.move_to_pose(H_b_ee, speed_factor=float(speed))

# === CAMERA (RealSense color) ===
class RealSenseColorCap:
    def __init__(self, serial, w, h, fps):
        self.serial = serial
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(str(serial))
        self.cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.profile = self.pipe.start(self.cfg)
        vsp = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        intr = vsp.get_intrinsics()
        self.fw, self.fh = intr.width, intr.height
        print(f"[realsense] Opened RGB S/N {serial}: {intr.width}x{intr.height}@{fps}")

    def read(self, timeout_ms=2000):
        t0 = time.time()
        while (time.time() - t0) * 1000 < timeout_ms:
            frames = self.pipe.poll_for_frames()
            if frames:
                color = frames.get_color_frame()
                if color:
                    return True, np.asanyarray(color.get_data())
            time.sleep(0.001)
        return False, None

    def stop(self):
        try: self.pipe.stop()
        except Exception: pass

# === AprilTag detector ===
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

# === Intrinsics loading / undist ===
def load_intrinsics():
    if os.path.isfile(INTRINSICS_FILE):
        data = np.load(INTRINSICS_FILE)
        K = data["K"].astype(np.float32)
        D = (data["D"] if "D" in data.files else data.get("dist", np.zeros(5))).astype(np.float32)
        print(f"[calib] Intrinsics: {INTRINSICS_FILE}")
        return K, D
    else:
        print(f"[calib] WARNING: Intrinsics file not found: {INTRINSICS_FILE}")
        return None, None

def undistort_and_params(gray, K, D):
    if K is None or D is None:
        vsp = rs.video_stream_profile(cap.profile.get_stream(rs.stream.color))
        intr = vsp.get_intrinsics()
        K2 = np.array([[intr.fx, 0, intr.ppx],
                       [0, intr.fy, intr.ppy],
                       [0, 0, 1]], np.float32)
        D2 = np.array(intr.coeffs[:5], np.float32)
        newK, _ = cv2.getOptimalNewCameraMatrix(K2, D2, gray.shape[::-1], 1.0)
        undist = cv2.undistort(gray, K2, D2, None, newK)
        cam_params = (float(newK[0,0]), float(newK[1,1]),
                      float(newK[0,2]), float(newK[1,2]))
        return undist, cam_params
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, gray.shape[::-1], 1.0)
        undist = cv2.undistort(gray, K, D, None, newK)
        cam_params = (float(newK[0,0]), float(newK[1,1]),
                      float(newK[0,2]), float(newK[1,2]))
        return undist, cam_params


# --- small SO(3) helpers (Euler ZYX: yaw, pitch, roll) ---
def euler_zyx_to_R(yaw, pitch, roll):
    cy, sy = np.cos(yaw),  np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], float)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], float)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], float)
    return Rz @ Ry @ Rx

def R_to_euler_zyx(R):
    # returns yaw, pitch, roll (Z-Y-X)
    # numerically stable for most robot poses
    pitch = -np.arcsin(np.clip(R[2,0], -1.0, 1.0))
    cp = np.cos(pitch)
    if abs(cp) < 1e-8:
        # gimbal-ish: fallback
        yaw  = np.arctan2(-R[0,1], R[1,1])
        roll = 0.0
    else:
        yaw  = np.arctan2(R[1,0], R[0,0])
        roll = np.arctan2(R[2,1], R[2,2])
    return yaw, pitch, roll

def apply_xyzrpy_offset(H_ref, dx, dy, dz, dyaw, dpitch, droll, deg=True):
    """Compose a target pose by translating in base and rotating RELATIVE to the start orientation."""
    H = H_ref.copy()
    # translation in base frame
    H[:3,3] = H[:3,3] + np.array([dx, dy, dz], float)
    # orientation: R_target = R_start * R_delta
    if deg:
        dyaw, dpitch, droll = np.deg2rad([dyaw, dpitch, droll])
    R_start = H_ref[:3,:3]
    R_delta = euler_zyx_to_R(dyaw, dpitch, droll)
    H[:3,:3] = R_start @ R_delta
    return H


def detect_any_tag(gray_undist, cam_params, tag_size_m=OBJ_TAG_SIZE, valid_ids=VALID_TAG_IDS):
    dets = detector.detect(gray_undist, estimate_tag_pose=True,
                           camera_params=cam_params, tag_size=tag_size_m)
    ok = False
    best = None
    for d in dets:
        if valid_ids and (d.tag_id not in valid_ids):
            continue
        area = cv2.contourArea(d.corners.astype(np.float32))
        if area < DETECT_MIN_AREA_PX:
            continue
        ok = True
        best = d
        break
    return ok, best, dets

def draw_overlay(bgr, det):
    cs = det.corners.astype(int)
    for k in range(4):
        cv2.line(bgr, tuple(cs[k]), tuple(cs[(k+1)%4]), (0,255,255), 2)
    # (axes omitted; we don't hold newK here)

# === Grid utilities ===
def generate_xy_offsets(nx, ny, dx, dy, order="raster"):
    offsets = []
    for j in range(ny):
        xs = list(range(nx))
        if order == "snake" and (j % 2 == 1):
            xs = xs[::-1]
        for i in xs:
            offsets.append((i*dx, j*dy))
    return offsets

def apply_xy_offset(H_b_ee_ref, dx, dy):
    H = H_b_ee_ref.copy()
    H[:3,3] = H[:3,3] + np.array([dx, dy, 0.0])
    return H

def apply_xyz_offset(H_b_ee_ref, dx, dy, dz):
    H = H_b_ee_ref.copy()
    H[:3,3] = H[:3,3] + np.array([dx, dy, dz], dtype=float)
    return H

# === Main ===
if __name__ == "__main__":
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump({
            "GRID_NX": GRID_NX, "GRID_NY": GRID_NY, "GRID_NZ": GRID_NZ,
            "GRID_DX": GRID_DX, "GRID_DY": GRID_DY, "GRID_DZ": GRID_DZ,
            "GRID_ORDER_XY": GRID_ORDER_XY, "Z_TOP_TO_BOTTOM": Z_TOP_TO_BOTTOM,
            "OBJ_TAG_SIZE_m": OBJ_TAG_SIZE,
            "SERIAL": SERIAL, "FRAME_W": FRAME_W, "FRAME_H": FRAME_H, "FPS": FPS
        }, f, indent=2)

    # 1) Connect camera
    cap = RealSenseColorCap(SERIAL, FRAME_W, FRAME_H, FPS)
    K, D = load_intrinsics()

    # 2) Read start pose as reference
    H_b_ee_start = get_pose_b_ee()
    # (KEEP_Z_ORIENT_FROM_START is already True; we keep Z and orientation)

    # 3) Build centered grid around the start pose
    # Centering offsets for each axis:
    x0 = - (GRID_NX - 1) / 2.0 * GRID_DX
    y0 = - (GRID_NY - 1) / 2.0 * GRID_DY
    z0 = - (GRID_NZ - 1) / 2.0 * GRID_DZ

    # XY visit order (list of (ox, oy))
    xy_offsets = generate_xy_offsets(GRID_NX, GRID_NY, GRID_DX, GRID_DY, order=GRID_ORDER_XY)

    # Z visiting order
    z_layers = list(range(GRID_NZ))
    if Z_TOP_TO_BOTTOM:
        z_layers = z_layers[::-1]

    # Total waypoints (include orientations)
    num_wp = GRID_NX * GRID_NY * GRID_NZ * len(ORI_LISTS)
    print(f"[grid] {num_wp} waypoints ({GRID_NX}x{GRID_NY}x{GRID_NZ} x {len(ORI_LISTS)} orientations)")

    saved_count, skipped_count = 0, 0
    idx = -1



    for kz in z_layers:
        dz = z0 + kz * GRID_DZ
        for (ox, oy) in xy_offsets:
            dx, dy = x0 + ox, y0 + oy
            for (yaw_deg, pitch_deg, roll_deg) in ORI_LISTS:
                idx += 1

                # --- sample Gaussian noise per axis (deg) ---
                nyaw = float(_rng.normal(0.0, ORI_NOISE_STD_DEG))
                npitch = float(_rng.normal(0.0, ORI_NOISE_STD_DEG))
                nroll = float(_rng.normal(0.0, ORI_NOISE_STD_DEG))

                # noisy commanded angles (deg)
                yaw_cmd_deg = _clamp_angle_deg(yaw_deg + nyaw, ORI_NOISE_CLAMP_DEG)
                pitch_cmd_deg = _clamp_angle_deg(pitch_deg + npitch, ORI_NOISE_CLAMP_DEG)
                roll_cmd_deg = _clamp_angle_deg(roll_deg + nroll, ORI_NOISE_CLAMP_DEG)

                # build target pose with NOISY angles (still relative to start orientation)
                H_target = apply_xyzrpy_offset(
                    H_b_ee_start, dx, dy, dz,
                    dyaw=yaw_cmd_deg, dpitch=pitch_cmd_deg, droll=roll_cmd_deg, deg=True
                )

                print(f"[move] {idx + 1}/{num_wp} -> "
                      f"Δx={dx:.3f} Δy={dy:.3f} Δz={dz:.3f} m | "
                      f"Δyaw={yaw_cmd_deg:+.1f} Δpitch={pitch_cmd_deg:+.1f} Δroll={roll_cmd_deg:+.1f} deg "
                      f"(nominal: {yaw_deg:+.1f}/{pitch_deg:+.1f}/{roll_deg:+.1f})")

                move_to_pose_b_ee(H_target, speed=0.2)
                time.sleep(APPROACH_PAUSE_S)

                # --- detect (unchanged) ---
                tag_ok = False
                last_frame_bgr = None
                last_gray_undist = None
                last_cam_params = None
                last_det = None
                for k in range(MAX_DETECT_FRAMES):
                    ok, frame = cap.read(timeout_ms=1500)
                    if not ok or frame is None:
                        continue
                    last_frame_bgr = frame.copy()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    undist, cam_params = undistort_and_params(gray, K, D)
                    last_gray_undist, last_cam_params = undist, cam_params

                    found, det, dets = detect_any_tag(undist, cam_params,
                                                      tag_size_m=OBJ_TAG_SIZE,
                                                      valid_ids=VALID_TAG_IDS)
                    if found:
                        tag_ok = True
                        last_det = det
                        break

                # --- always capture the actually reached robot pose now ---
                H_now = get_pose_b_ee()
                p_now = H_now[:3, 3].tolist()
                yaw_now, pitch_now, roll_now = [float(x) for x in R_to_euler_zyx(H_now[:3, :3])]
                yaw_now_deg, pitch_now_deg, roll_now_deg = np.rad2deg([yaw_now, pitch_now, roll_now]).tolist()

                base_name = (f"grid_{idx:04d}_dx{dx:+.3f}_dy{dy:+.3f}_dz{dz:+.3f}_"
                             f"yaw{yaw_cmd_deg:+.1f}_pit{pitch_cmd_deg:+.1f}_rol{roll_cmd_deg:+.1f}")

                if tag_ok:
                    cv2.imwrite(os.path.join(RUN_DIR, base_name + "_raw.png"), last_frame_bgr)
                    if SAVE_OVERLAY:
                        vis = cv2.cvtColor(last_gray_undist, cv2.COLOR_GRAY2BGR)
                        cs = last_det.corners.astype(int)
                        for t in range(4):
                            cv2.line(vis, tuple(cs[t]), tuple(cs[(t + 1) % 4]), (0, 255, 255), 2)
                        cv2.imwrite(os.path.join(RUN_DIR, base_name + "_undist_overlay.png"), vis)

                    meta = {
                        "idx": idx,
                        "dx": float(dx), "dy": float(dy), "dz": float(dz),

                        # commanded (NOISY) angles used to build H_target
                        "yaw_cmd_deg": float(yaw_cmd_deg),
                        "pitch_cmd_deg": float(pitch_cmd_deg),
                        "roll_cmd_deg": float(roll_cmd_deg),

                        # nominal grid angles before noise (for analysis/repro)
                        "yaw_nominal_deg": float(yaw_deg),
                        "pitch_nominal_deg": float(pitch_deg),
                        "roll_nominal_deg": float(roll_deg),

                        "tag_id": int(last_det.tag_id),
                        "cam_params": {"fx": last_cam_params[0], "fy": last_cam_params[1],
                                       "cx": last_cam_params[2], "cy": last_cam_params[3]},

                        # robot pose logging (actual reached)
                        "H_b_ee_4x4": H_now.tolist(),
                        "pos_m": {"x": p_now[0], "y": p_now[1], "z": p_now[2]},
                        "euler_zyx_deg": {"yaw": yaw_now_deg, "pitch": pitch_now_deg, "roll": roll_now_deg}
                    }

                    with open(os.path.join(RUN_DIR, base_name + ".json"), "w") as f:
                        json.dump(meta, f, indent=2)

                    saved_count += 1
                    print(f"[save] {base_name} (tag {last_det.tag_id})")
                else:
                    skipped_count += 1
                    print(f"[skip]  {base_name} (no valid tag detected)")

    print(f"[done] saved: {saved_count}, skipped: {skipped_count}, folder: {RUN_DIR}")

    # Cleanup
    cap.stop()
