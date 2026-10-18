"""Head camera <-> robot pelvis frame: pixel + depth -> 3D target, and 3D -> pixel for overlays.

Chain: pixel (u, v) + aligned depth -> point in the camera's optical frame (pinhole intrinsics)
-> d435_link (fixed optical rotation) -> torso_link (URDF mount) -> pelvis (live waist angles,
T_pelvis_torso from arm_server). Pure numpy; fetching frames lives in server.py.

Mount from Unitree's g1_body29_hand14.urdf, joint d435_joint (torso_link -> d435_link):
xyz (0.0576235, 0.01753, 0.42987), rpy (0, 0.8307767, 0): 43 cm up, pitched 47.6 deg down.
Michael's measured camera-pose-head.json (on the Jetson) may refine this; T_TORSO_CAM can be swapped.
"""
from __future__ import annotations

import numpy as np


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
    return T


# Optical frame (x right, y down, z out of the lens) -> camera link (x forward, y left, z up).
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_LINK_OPTICAL = np.eye(4)
T_LINK_OPTICAL[:3, :3] = R_LINK_OPTICAL
T_TORSO_LINK = transform((0.0576235, 0.01753, 0.42987), (0.0, 0.8307767239493009, 0.0))
T_TORSO_CAM = T_TORSO_LINK @ T_LINK_OPTICAL   # optical frame in torso coordinates

DEPTH_RANGE = (0.10, 3.0)   # metres; outside this the reading is noise or missing


def depth_at(depth_m: np.ndarray, u: float, v: float, radius: int = 3) -> tuple[float | None, float]:
    """Median of valid depths in a (2r+1)^2 window around (u, v). Returns (metres or None, valid fraction).
    The median survives holes (0) and object edges far better than a single pixel."""
    h, w = depth_m.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None, 0.0
    win = depth_m[max(0, vi - radius):vi + radius + 1, max(0, ui - radius):ui + radius + 1]
    ok = win[(win > DEPTH_RANGE[0]) & (win < DEPTH_RANGE[1])]
    frac = ok.size / win.size
    if not ok.size and radius < 9:   # dark/shiny surfaces leave holes: look a little wider before giving up
        z, _ = depth_at(depth_m, u, v, radius * 3)
        return z, 0.0
    return (float(np.median(ok)) if ok.size else None), frac


def deproject(u: float, v: float, z: float, K: dict) -> np.ndarray:
    """Pixel + depth (m) -> point in the optical frame. K: fx, fy, ppx, ppy (zero distortion on this camera)."""
    return np.array([(u - K["ppx"]) / K["fx"] * z, (v - K["ppy"]) / K["fy"] * z, z])


def project(p_opt: np.ndarray, K: dict) -> tuple[float, float] | None:
    """Optical-frame point -> pixel, or None if it's behind the camera."""
    if p_opt[2] <= 1e-6:
        return None
    return K["fx"] * p_opt[0] / p_opt[2] + K["ppx"], K["fy"] * p_opt[1] / p_opt[2] + K["ppy"]


def pelvis_from_cam(T_pelvis_torso: np.ndarray, T_torso_cam: np.ndarray = T_TORSO_CAM) -> np.ndarray:
    return np.asarray(T_pelvis_torso, float) @ T_torso_cam


def pixel_to_pelvis(u, v, z, K, T_pelvis_cam) -> np.ndarray:
    p = deproject(u, v, z, K)
    return (T_pelvis_cam @ np.append(p, 1.0))[:3]


def pelvis_to_pixel(p_pelvis, K, T_pelvis_cam):
    p = np.linalg.inv(T_pelvis_cam) @ np.append(np.asarray(p_pelvis, float), 1.0)
    return project(p[:3], K)


def to_pixels(u: float, v: float, scale: str, width: int, height: int) -> tuple[float, float]:
    """Accept pixels, or normalized coordinates as VLMs return them: 'norm1000' (0-1000, Gemini Robotics-ER style)
    or 'norm1' (0-1)."""
    if scale == "pixels":
        return float(u), float(v)
    if scale == "norm1000":
        return u / 1000.0 * width, v / 1000.0 * height
    if scale == "norm1":
        return u * width, v * height
    raise ValueError("scale must be pixels, norm1000 or norm1")


def segment(depth_m, u, v, K, T_pelvis_cam, desk_z, min_h=0.008, grow_px=200, snap_px=45, max_r=0.12, step_m=0.012, box=None):
    """The object under pixel (u, v): pixels standing > min_h above the desk, flood-filled from the click (4-connected,
    neighbours < step_m apart in 3D so touching objects split at depth edges, within max_r of the seed in xy). Returns pelvis-frame geometry so an agent only has to point:
      centre (xy centroid, z mid-height), top_z, height, length/width (PCA extents in xy), jaw_deg (the axis the
      jaws should close along = the narrow direction), ring (hollow centre, like a tape roll), pixels.
    box: [u1, v1, u2, v2] from a detector / VLM (bbox_2d): only pixels inside it count, so touching objects split
      cleanly; (u, v) is then just the seed (pass the box centre).
    None if nothing stands above the desk there."""
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth_m
    ok = (z > DEPTH_RANGE[0]) & (z < DEPTH_RANGE[1])
    pts = np.stack([(uu - K["ppx"]) / K["fx"] * z, (vv - K["ppy"]) / K["fy"] * z, z, np.ones_like(z)], -1)
    P = pts @ T_pelvis_cam.T
    above = ok & (P[..., 2] > desk_z + min_h)
    if box is not None:
        u1, v1, u2, v2 = (int(round(c)) for c in box)
        inside = np.zeros_like(above)
        inside[max(0, v1):max(0, v2) + 1, max(0, u1):max(0, u2) + 1] = True
        above &= inside
        max_r, grow_px = 1.0, 10 ** 6
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    if not above[vi, ui]:                      # the click grazed an edge or a hole: snap to the nearest object pixel
        y0, x0 = max(0, vi - snap_px), max(0, ui - snap_px)    # (a ring's centre is a hole: reach its wall)
        ys, xs = np.nonzero(above[y0:vi + snap_px + 1, x0:ui + snap_px + 1])
        if not len(ys):
            return None
        k = np.argmin((ys + y0 - vi) ** 2 + (xs + x0 - ui) ** 2)
        vi, ui = ys[k] + y0, xs[k] + x0
    seen = np.zeros_like(above)

    def flood(sy, sx):
        seen[sy, sx] = True
        stack, region = [(sy, sx)], []
        while stack:
            y, x = stack.pop()
            region.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and above[ny, nx] and abs(ny - sy) + abs(nx - sx) < grow_px
                        and np.linalg.norm(P[ny, nx, :3] - P[y, x, :3]) < step_m
                        and np.linalg.norm(P[ny, nx, :2] - P[sy, sx, :2]) < max_r):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        return region

    region = flood(vi, ui)
    if box is not None:      # in a box: the biggest piece is the object (a ring's wall can break into arcs; merge them
        pieces = [region] + [flood(y, x) for y, x in zip(*np.nonzero(above)) if not seen[y, x]]
        pieces.sort(key=len, reverse=True)
        region = [px for pc in pieces if len(pc) > 0.15 * len(pieces[0]) for px in pc]    # ...and keep the big arcs)
    ys, xs = np.array(region).T
    p = P[ys, xs, :3]
    xy = p[:, :2]
    c = xy.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((xy - c).T))
    minor, major = evecs[:, 0], evecs[:, 1]
    proj_min, proj_maj = (xy - c) @ minor, (xy - c) @ major
    top = float(np.percentile(p[:, 2], 95))
    width = float(np.ptp(proj_min))
    length = float(np.ptp(proj_maj))
    centre_xy = c + minor * (proj_min.max() + proj_min.min()) / 2 + major * (proj_maj.max() + proj_maj.min()) / 2
    # hollow: the middle of the footprint is (near) desk height, e.g. a tape roll or an empty cup seen from above
    near = np.linalg.norm(xy - centre_xy, axis=1) < 0.25 * min(width, length)
    ring = bool(min(width, length) > 0.05 and near.sum() < 0.02 * len(xy))
    return {"centre": [float(centre_xy[0]), float(centre_xy[1]), (top + desk_z) / 2], "top_z": top,
            "height": top - desk_z, "length": length, "width": width,
            "jaw_deg": float(np.degrees(np.arctan2(minor[1], minor[0]))), "ring": ring, "pixels": len(region),
            "box_min": [float(xy[:, 0].min()), float(xy[:, 1].min()), desk_z], "box_max": [float(xy[:, 0].max()), float(xy[:, 1].max()), top]}


if __name__ == "__main__":
    # Self-check: round trips, the mount's direction, and the robust depth window.
    K = {"fx": 604.93, "fy": 604.43, "ppx": 326.58, "ppy": 247.79}
    T = pelvis_from_cam(np.eye(4))
    for u, v, z in ((326.58, 247.79, 0.8), (100, 400, 0.6), (600, 50, 1.5)):
        p = pixel_to_pelvis(u, v, z, K, T)
        uu, vv = pelvis_to_pixel(p, K, T)
        assert abs(uu - u) < 1e-6 and abs(vv - v) < 1e-6, (u, v, uu, vv)
    # Image centre at 0.8 m: the camera looks forward and 47.6 deg down, so the point is ahead and below the lens.
    p = pixel_to_pelvis(K["ppx"], K["ppy"], 0.8, K, T)
    lens = T[:3, 3]
    d = (p - lens) / 0.8
    assert d[0] > 0.6 and d[2] < -0.6 and abs(d[1]) < 1e-9, d
    assert abs(np.degrees(np.arctan2(-d[2], d[0])) - 47.6) < 0.1
    # A pixel left of centre maps to +y (robot's left); below centre maps lower and nearer.
    pl = pixel_to_pelvis(100, K["ppy"], 0.8, K, T)
    assert pl[1] > p[1]
    # Depth window: holes ignored, median robust to an edge.
    dm = np.full((480, 640), 0.9)
    dm[242:246, 322:326] = 0.0          # partial hole
    dm[244:248, 325:328] = 2.5          # far edge poking into the window
    z, frac = depth_at(dm, 324, 244)
    assert z == 0.9 and 0 < frac < 1, (z, frac)
    dm[236:252, 316:332] = 0.0          # hole bigger than the 7x7 window: the widened 19x19 search still answers
    z, frac = depth_at(dm, 324, 244)
    assert z is not None and frac == 0.0, (z, frac)
    assert depth_at(dm, 700, 10) == (None, 0.0)
    assert to_pixels(500, 500, "norm1000", 640, 480) == (320.0, 240.0)
    # Segment a synthetic 10 x 4 cm box (3 cm tall, long axis along y) on the desk, seen through the real mount.
    desk = -0.06
    dm = np.zeros((480, 640))
    Tinv = np.linalg.inv(T)
    for vi in range(480):              # ray-cast each pixel against the desk plane and the box top
        for ui in range(0, 640):
            ray = T[:3, :3] @ np.array([(ui - K["ppx"]) / K["fx"], (vi - K["ppy"]) / K["fy"], 1.0])
            for zp, inside in ((desk + 0.03, lambda q: abs(q[0] - 0.30) < 0.02 and abs(q[1]) < 0.05), (desk, lambda q: True)):
                t = (zp - lens[2]) / ray[2]
                q = lens + t * ray
                if t > 0 and inside(q):
                    dm[vi, ui] = (Tinv @ np.append(q, 1))[2]
                    break
    u0, v0 = pelvis_to_pixel([0.30, 0.0, desk + 0.03], K, T)
    g = segment(dm, u0, v0, K, T, desk)
    assert abs(g["width"] - 0.04) < 0.006 and abs(g["length"] - 0.10) < 0.006, g
    assert abs(g["height"] - 0.03) < 0.003, g
    assert min(abs(g["jaw_deg"]) % 180, 180 - abs(g["jaw_deg"]) % 180) < 3, g     # jaws close along x, across the 4 cm
    assert abs(g["centre"][0] - 0.30) < 0.005 and abs(g["centre"][1]) < 0.005 and not g["ring"], g
    print("vision self-check ok; lens at", lens.round(3), "pelvis->torso identity")
