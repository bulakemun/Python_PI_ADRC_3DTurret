"""
3D visualization of the turret, its line of sight and the target board, plus a
turret-mounted "camera POV" view.

Two renderers:
  1. ``draw_world_view`` -- a 3D world view showing the static base, the barrel
     orientation / line-of-sight ray, the nominal sight-line to the board, and
     the target board at range.
  2. ``draw_pov_view``   -- a 2D reprojection of the scene from the turret's
     current azimuth/elevation (a pinhole camera looking along the barrel),
     showing the board drift toward centre-frame as the controller converges.

Both are pure functions: they take a Matplotlib axis plus the current
``TurretModel``/``TargetBoard`` state and draw into the axis. No control or
plant logic lives here.
"""

from __future__ import annotations

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


def _camera_frame(direction: np.ndarray):
    """Right/up/forward orthonormal basis for a camera looking along ``direction``."""
    forward = direction / (np.linalg.norm(direction) + 1e-12)
    # Guard against forward nearly parallel to world up (straight up/down).
    up_ref = WORLD_UP
    if abs(np.dot(forward, up_ref)) > 0.999:
        up_ref = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_ref)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    return right, up, forward


def draw_world_view(ax, turret_model, target_board, los_length=None) -> None:
    """Render the world view into a 3D Matplotlib axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()

    base = turret_model.base_position
    board_center = target_board.position
    if los_length is None:
        los_length = float(np.linalg.norm(board_center - base) * 1.1)

    # Target board face.
    corners = target_board.corners()
    board = Poly3DCollection(
        [corners], facecolor="tab:red", edgecolor="darkred", alpha=0.5
    )
    ax.add_collection3d(board)

    # Nominal sight-line from base to board centre (dashed reference).
    ax.plot(
        [base[0], board_center[0]],
        [base[1], board_center[1]],
        [base[2], board_center[2]],
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="nominal LOS",
    )

    # Actual barrel line of sight.
    tip = base + turret_model.barrel_direction * los_length
    ax.plot(
        [base[0], tip[0]],
        [base[1], tip[1]],
        [base[2], tip[2]],
        color="tab:blue",
        linewidth=2.0,
        label="barrel LOS",
    )

    # Turret base marker.
    ax.scatter(*base, color="black", s=40, label="turret")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    span = los_length
    ax.set_xlim(0, span)
    ax.set_ylim(-span / 2, span / 2)
    ax.set_zlim(-span / 4, span / 4)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("World view")


def draw_pov_view(ax, turret_model, target_board, fov_deg=40.0) -> None:
    """Render the turret camera POV into a 2D Matplotlib axis."""
    ax.clear()

    right, up, forward = _camera_frame(turret_model.barrel_direction)
    origin = turret_model.base_position
    half_extent = np.tan(np.radians(fov_deg) / 2.0)

    # Project each board corner through a pinhole camera looking along forward.
    corners = target_board.corners()
    projected = []
    all_in_front = True
    for p in corners:
        v = p - origin
        z = np.dot(v, forward)
        if z <= 1e-6:
            all_in_front = False
            break
        projected.append((np.dot(v, right) / z, np.dot(v, up) / z))

    if all_in_front and projected:
        poly = plt_polygon(projected)
        ax.add_patch(poly)

    # Cross-hair at frame centre (the barrel's aim point).
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.axvline(0.0, color="gray", linewidth=0.6)
    ax.plot(0.0, 0.0, marker="+", color="black", markersize=10)

    ax.set_xlim(half_extent, -half_extent)  # camera x: right is +, flip for screen
    ax.set_ylim(-half_extent, half_extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Turret POV")


def plt_polygon(points):
    """Small helper: build a filled polygon patch from image-plane points."""
    from matplotlib.patches import Polygon

    return Polygon(
        points, closed=True, facecolor="tab:red", edgecolor="darkred", alpha=0.6
    )
