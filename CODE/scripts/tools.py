import numpy as np

def rot_mat_to_euler_zyx(r, degrees=False):
    """
    Decompose a 3x3 rotation matrix R into ZYX Euler angles:
    yaw (about Z), pitch (about Y), roll (about X).
    Returns (yaw, pitch, roll).
    """
    r= np.asarray(r, dtype=float)
    assert r.shape == (3, 3), "R must be 3x3"

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # Stable extraction with gimbal-lock handling:
    if abs(r[2, 0]) < 1.0 - 1e-8:
        pitch = np.arcsin(-r[2, 0])
        roll  = np.arctan2(r[2, 1], r[2, 2])
        yaw   = np.arctan2(r[1, 0], r[0, 0])
    else:
        # Gimbal lock: pitch ≈ ±pi/2, roll and yaw are coupled
        pitch = np.pi/2 * np.sign(-r[2, 0])
        roll  = 0.0
        # choose yaw that matches R as best as possible
        yaw   = np.arctan2(-r[0, 1], r[1, 1])

    if degrees:
        return tuple(np.degrees([yaw, pitch, roll]))
    return yaw, pitch, roll

def to_H(R, t):
    H = np.eye(4); H[:3,:3] = R; H[:3,3] = t.reshape(3)
    return H

def inv_H(H):
    Ri = H[:3,:3].T
    ti = -Ri @ H[:3,3]
    Hi = np.eye(4); Hi[:3,:3] = Ri; Hi[:3,3] = ti
    return Hi

def H_to_xyzrpy_ZYX(H):
    R = H[:3,:3]; x,y,z = H[:3,3]
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-9:
        yaw   = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = np.arctan2(R[2,1], R[2,2])
    else:  # near gimbal lock
        yaw   = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        roll  = 0.0
    return np.array([x,y,z, roll,pitch,yaw])  # (x,y,z,r,p,y)

def se3_blend(H_new, H_old, alpha):
    """Blend two poses (translation linear, rotation via SVD). alpha = weight on NEW."""
    if H_old is None: return H_new
    t_old, t_new = H_old[:3,3], H_new[:3,3]
    t_out = (1.0 - alpha) * t_old + alpha * t_new
    R_old, R_new = H_old[:3,:3], H_new[:3,:3]
    M = (1.0 - alpha) * R_old + alpha * R_new
    U, _, Vt = np.linalg.svd(M)
    R_out = U @ Vt
    H_out = np.eye(4); H_out[:3,:3] = R_out; H_out[:3,3] = t_out
    return H_out


def rot_geodesic_angle_deg(R1, R2):
    """Smallest angle between two rotations (degrees)."""
    R = R1.T @ R2
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))



def pose_delta(Ha, Hb):
    """Return (pos_err_m, ang_err_deg) between two world poses."""
    pa, pb = Ha[:3,3], Hb[:3,3]
    print("calculating pose delta. this is pa and pb:", pa,pb)
    Ra, Rb = Ha[:3,:3], Hb[:3,:3]
    pos_err = np.linalg.norm(pa - pb )
    ang_err = rot_geodesic_angle_deg(Ra, Rb)
    return pos_err, ang_err



