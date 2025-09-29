# single_cam_apriltag_pose_base.py
import os, sys, numpy as np, cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from tools import to_H, inv_H, rot_mat_to_euler_zyx, pose_delta  # your helpers
import panda_py

# ======= CONFIG =======
SERIAL = "839112062097"               # your single camera
FRAME_W, FRAME_H, FPS = 1280, 720, 30
OBJ_TAG_SIZE = 0.023                  # meters (outer black border)
OBJ_TAG_IDS = {2, 3}                  # track these tag IDs (edit as needed)

DATA_ROOT = "DATA"                    # NOTE: no leading "../" per your layout
INTRINSICS_FILE = os.path.join(DATA_ROOT, f"Camera_intrinsics_{SERIAL}_Res{FRAME_W}x{FRAME_H}.npz")
HAND_EYE_NPZ    = os.path.join(DATA_ROOT, f"CAM_{SERIAL}_calibration", "handeye_results",
                               "2025-09-21T16-00-48_handeye_839112062097_PARK.npz")
FIXED_BTc_NPY   = os.path.join(DATA_ROOT, f"H_base_cam_{SERIAL}.npy")  # optional (preferred if present)

ROBOT_IP = "172.16.0.2"
ROBOT_POSE_IS_TOOL_IN_WORLD = False   # flip if your API returns TOOL->BASE

# ======= Robot =======
try:
    robot = panda_py.Panda(hostname=ROBOT_IP)
    print(f"[robot] Connected to Panda at {ROBOT_IP}")
except Exception as e:
    robot = None
    print(f"[robot] Could not connect to Panda: {e}")

def get_base_to_tool():
    if robot is None:
        return None
    H = np.array(robot.get_pose(), dtype=float)  # typical: ^bT_ee
    if ROBOT_POSE_IS_TOOL_IN_WORLD:
        H = inv_H(H)
    return H

# ======= AprilTag detector =======
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

# ======= RealSense wrapper (color) =======
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
        print(f"Opened RealSense RGB S/N {serial}: {intr.width}x{intr.height}@{fps}")

    def read(self):
        frames = self.pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color: return False, None
        return True, np.asanyarray(color.get_data())

    def release(self):
        try: self.pipe.stop()
        except Exception: pass

# ======= Calibration loading =======
def load_intrinsics():
    K, D = None, None
    if os.path.isfile(INTRINSICS_FILE):
        data = np.load(INTRINSICS_FILE)
        K = data["K"].astype(np.float32)
        D = (data["D"] if "D" in data.files else data.get("dist", np.zeros(5))).astype(np.float32)
        print(f"[calib] Intrinsics: {INTRINSICS_FILE}")
    else:
        print(f"[calib] WARNING: Intrinsics file not found: {INTRINSICS_FILE}")
    return K, D

def load_fixed_bTc():
    if os.path.isfile(FIXED_BTc_NPY):
        H = np.load(FIXED_BTc_NPY)
        if H.shape == (4,4):
            print(f"[calib] Fixed ^bT_c: {FIXED_BTc_NPY}")
            print(H)
            return H
    return None

def load_handeye_gTc():
    if os.path.isfile(HAND_EYE_NPZ):
        with np.load(HAND_EYE_NPZ) as data:
            key = "H_g_c" if "H_g_c" in data.files else ("Hgc" if "Hgc" in data.files else None)
            if key is None:
                print(f"[calib] {HAND_EYE_NPZ} has no H_g_c key")
                return None
            H = np.array(data[key], dtype=float)
            if H.shape == (4,4):
                print(f"[calib] Hand-eye (^gT_c) from {HAND_EYE_NPZ}")
                return H
    else:
        print(f"[calib] Hand-eye NPZ not found: {HAND_EYE_NPZ}")
    return None

# ======= Drawing =======
def draw_axes(img, K, dist, rvec, tvec, length=0.5*OBJ_TAG_SIZE):
    axis = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    p0 = tuple(imgpts[0].ravel().astype(int))
    cv2.line(img, p0, tuple(imgpts[1].ravel().astype(int)), (0,0,255), 2)
    cv2.line(img, p0, tuple(imgpts[2].ravel().astype(int)), (0,255,0), 2)
    cv2.line(img, p0, tuple(imgpts[3].ravel().astype(int)), (255,0,0), 2)

# ======= Init single camera =======
cap = RealSenseColorCap(SERIAL, FRAME_W, FRAME_H, FPS)
K, D = load_intrinsics()
H_base_cam_fixed = load_fixed_bTc()
H_g_c = load_handeye_gTc()

if H_base_cam_fixed is None and H_g_c is None:
    print("[calib] No ^bT_c or ^gT_c available. "
          "Provide DATA/H_base_cam_<SN>.npy or the hand-eye NPZ, "
          "or connect the robot to compute ^bT_c live.")
warned_no_extr = False
autosaved_bTc = False

# ======= Logging buffer (optional) =======
TMAX = 10000
log = np.zeros((TMAX, 1, 1 + max(OBJ_TAG_IDS), 6))
cycle = -1

print("Press 'q' or ESC to quit.")




while True:

    cycle += 1
    slot = cycle % TMAX

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[warn] grab failed")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use file intrinsics if available; else fall back to RS intrinsics (identity distortion)
    if K is None or D is None:
        vsp = rs.video_stream_profile(cap.profile.get_stream(rs.stream.color))
        intr = vsp.get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], np.float32)
        D = np.array(intr.coeffs[:5], np.float32)

    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, gray.shape[::-1], 1.0)
    undist = cv2.undistort(gray, K, D, None, newK)
    vis = cv2.cvtColor(undist, cv2.COLOR_GRAY2BGR)

    # Determine ^bT_c for this frame
    H_b_c = H_base_cam_fixed
    if H_b_c is None and H_g_c is not None:
        H_b_g = get_base_to_tool()
        if H_b_g is not None:
            H_b_c = H_b_g @ H_g_c
            # Optional: persist fixed extrinsic once (so next run doesn't need robot)
            if not autosaved_bTc:
                try:
                    np.save(FIXED_BTc_NPY, H_b_c)
                    print(f"[calib] Auto-saved ^bT_c to {FIXED_BTc_NPY}")
                    autosaved_bTc = True
                    H_base_cam_fixed = H_b_c  # adopt as fixed
                except Exception as e:
                    print(f"[calib] Could not save {FIXED_BTc_NPY}: {e}")

    if H_b_c is None:
        if not warned_no_extr:
            print("[extr] No ^bT_c available. Need DATA/H_base_cam_<SN>.npy or robot+hand-eye.")
            warned_no_extr = True
        cv2.putText(vis, "No extrinsics (^bT_c).", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("AprilTag Pose (cam)", vis)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
        continue

    # Detect object tags
    dets = detector.detect(
        undist,
        estimate_tag_pose=True,
        camera_params=(newK[0,0], newK[1,1], newK[0,2], newK[1,2]),
        tag_size=OBJ_TAG_SIZE
    )

    for det in dets:
        if det.tag_id not in OBJ_TAG_IDS:
            continue
        R_c_o = np.array(det.pose_R, dtype=float)
        t_c_o = np.array(det.pose_t, dtype=float).reshape(3,1)
        rvec, _ = cv2.Rodrigues(R_c_o)

        # Draw tag + axes
        cs = det.corners.astype(int)
        for k in range(4):
            cv2.line(vis, tuple(cs[k]), tuple(cs[(k+1)%4]), (0,255,255), 2)
        draw_axes(vis, newK, np.zeros(5), rvec, t_c_o)
        cv2.putText(vis, f"TAG {det.tag_id}", (10, 30 + 30*det.tag_id),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        # Camera->object, then Base->object
        H_c_o = to_H(R_c_o, t_c_o)
        H_b_o = H_b_c @ H_c_o

        wx, wy, wz = H_b_o[:3,3]
        yaw, pitch, roll = rot_mat_to_euler_zyx(H_b_o[:3,:3], True)
        log[slot, 0, det.tag_id, :] = [wx, wy, wz, yaw, pitch, roll]

        cx, cy, cz = t_c_o.ravel()
        cv2.putText(vis,
            f"CAM[x:{cx:.3f} y:{cy:.3f} z:{cz:.3f}]  BASE[x:{wx:.3f} y:{wy:.3f} z:{wz:.3f}]",
            (10, 60 + 30*det.tag_id), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,200,50), 2
        )

        # Compare to robot tool pose if connected
        H_b_ee = get_base_to_tool()
        if H_b_ee is not None:
            pos_err_m, ang_err_deg = pose_delta(H_b_ee, H_b_o)
            cv2.putText(vis,
                f"ERR to tool: {pos_err_m*1000:.1f} mm, {ang_err_deg:.2f} deg",
                (10, 90 + 30*det.tag_id),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,180,255), 2
            )
        else:
            cv2.putText(vis, "Robot not connected",
                        (10, 90 + 30*det.tag_id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

    cv2.imshow("AprilTag Pose (cam)", vis)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
