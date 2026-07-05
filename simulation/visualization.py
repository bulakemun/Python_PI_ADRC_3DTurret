"""
Real-time 3D visualization of the turret, the ground, and the target board,
rendered with PyVista (VTK).

Two renderers share the same simulation state:
  1. World view -- a shaded 3D turret model (static base, a platform that yaws
     with azimuth, and a barrel that pitches with elevation), a ground plane,
     and the bullseye target board downrange.
  2. Turret POV -- a camera placed at the barrel pivot looking along the barrel,
     so the target board slides toward frame centre as the controller converges.

This module builds the meshes/actors and updates them each frame (actor
transforms + POV camera). It owns no control or plant logic: it is handed the
current ``TurretModel`` / ``TargetBoard`` state and reflects it. Persistent
actor state (a class) is used because VTK actors live across frames, exactly
the case CLAUDE.md allows classes for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyvista as pv

#: Height of the barrel pivot above the ground (metres).
PIVOT_HEIGHT = 1.2
#: Barrel length (metres).
BARREL_LENGTH = 2.4

# Palette (kept here so both views stay consistent).
_COL_BASE = "#3a3f4b"
_COL_PLATFORM = "#5b6472"
_COL_BARREL = "#8a929e"
_COL_MUZZLE = "#c9ced6"
_COL_GROUND = "#2c3e34"
_COL_BOARD_OUTER = "#c0392b"
_COL_BOARD_MID = "#ecf0f1"
_COL_BOARD_BULL = "#c0392b"


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _matrix(rot: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 rotation + translation."""
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = translation
    return m


def yaw_transform(azimuth: float, pivot: np.ndarray) -> np.ndarray:
    """Transform for parts that yaw with azimuth only (the platform)."""
    return _matrix(_rot_z(azimuth), pivot)


def barrel_transform(azimuth: float, elevation: float, pivot: np.ndarray) -> np.ndarray:
    """Transform mapping the local barrel frame to the world.

    ``R = Rz(azimuth) @ Ry(-elevation)`` sends local +x to the barrel
    direction ``(cos el cos az, cos el sin az, sin el)``.
    """
    return _matrix(_rot_z(azimuth) @ _rot_y(-elevation), pivot)


def _turret_platform_mesh() -> pv.PolyData:
    """Yaw platform + trunnion yoke, in the local pivot frame (pivot at origin)."""
    platform = pv.Cylinder(
        center=(0, 0, -0.15), direction=(0, 0, 1), radius=0.55, height=0.3
    )
    yoke_l = pv.Box(bounds=(-0.25, 0.25, 0.45, 0.62, -0.1, 0.55))
    yoke_r = pv.Box(bounds=(-0.25, 0.25, -0.62, -0.45, -0.1, 0.55))
    return platform + yoke_l + yoke_r


def _turret_barrel_mesh() -> pv.PolyData:
    """Barrel + muzzle + breech, in the local pivot frame (fires along +x)."""
    barrel = pv.Cylinder(
        center=(BARREL_LENGTH / 2, 0, 0), direction=(1, 0, 0),
        radius=0.13, height=BARREL_LENGTH,
    )
    muzzle = pv.Cylinder(
        center=(BARREL_LENGTH, 0, 0), direction=(1, 0, 0), radius=0.19, height=0.18
    )
    breech = pv.Box(bounds=(-0.45, 0.15, -0.24, 0.24, -0.24, 0.24))
    trunnion = pv.Cylinder(
        center=(0, 0, 0), direction=(0, 1, 0), radius=0.16, height=1.2
    )
    return barrel + muzzle + breech + trunnion


def _board_actors_meshes(board):
    """Bullseye rings for the target board, facing back toward the turret (-x)."""
    center = board.position
    normal = (1.0, 0.0, 0.0)
    r = min(board.width, board.height_dim) / 2.0
    # Nudge each ring slightly toward the turret so they don't z-fight.
    outer = pv.Disc(center=center - np.array([0.05, 0, 0]), inner=0.0,
                    outer=r, normal=normal, c_res=48)
    mid = pv.Disc(center=center - np.array([0.10, 0, 0]), inner=0.0,
                  outer=r * 0.62, normal=normal, c_res=48)
    bull = pv.Disc(center=center - np.array([0.15, 0, 0]), inner=0.0,
                   outer=r * 0.25, normal=normal, c_res=48)
    return [
        (outer, _COL_BOARD_OUTER),
        (mid, _COL_BOARD_MID),
        (bull, _COL_BOARD_BULL),
    ]


def _add_environment(plotter, board, ground_size: float = 60.0) -> None:
    """Add ground plane + target board to the *currently active* subplot."""
    ground = pv.Plane(
        center=(ground_size / 2 - 5, 0, 0), direction=(0, 0, 1),
        i_size=ground_size, j_size=ground_size, i_resolution=24, j_resolution=24,
    )
    plotter.add_mesh(ground, color=_COL_GROUND, ambient=0.3, show_edges=True,
                     edge_color="#24322a", line_width=1)
    for mesh, color in _board_actors_meshes(board):
        plotter.add_mesh(mesh, color=color, ambient=0.35, smooth_shading=True)


@dataclass
class TurretScene:
    """Holds the world-view actors that move each frame and the pivot point."""

    pivot: np.ndarray
    platform_actor: object = field(default=None)
    barrel_actor: object = field(default=None)

    def update(self, turret) -> None:
        """Reflect the turret's current azimuth/elevation into the actors."""
        az, el = turret.orientation
        if self.platform_actor is not None:
            self.platform_actor.user_matrix = yaw_transform(az, self.pivot)
        if self.barrel_actor is not None:
            self.barrel_actor.user_matrix = barrel_transform(az, el, self.pivot)


def build_world_view(plotter, turret, board) -> TurretScene:
    """Build the world-view subplot (call after selecting it). Returns a scene."""
    pivot = turret.base_position + np.array([0.0, 0.0, PIVOT_HEIGHT])

    _add_environment(plotter, board)

    # Static base column.
    base = pv.Cylinder(
        center=turret.base_position + np.array([0, 0, PIVOT_HEIGHT / 2]),
        direction=(0, 0, 1), radius=0.5, height=PIVOT_HEIGHT,
    )
    plotter.add_mesh(base, color=_COL_BASE, ambient=0.25, smooth_shading=True)

    platform_actor = plotter.add_mesh(
        _turret_platform_mesh(), color=_COL_PLATFORM, ambient=0.25,
        smooth_shading=True,
    )
    barrel_actor = plotter.add_mesh(
        _turret_barrel_mesh(), color=_COL_BARREL, metallic=0.4, roughness=0.5,
        smooth_shading=True,
    )
    scene = TurretScene(pivot=pivot, platform_actor=platform_actor,
                        barrel_actor=barrel_actor)
    scene.update(turret)

    plotter.add_axes()
    plotter.set_background("#1b2733", top="#33475b")
    plotter.camera_position = [
        (10.0, -11.0, 6.5),  # eye
        (0.0, 0.0, PIVOT_HEIGHT),  # focus on the turret
        (0.0, 0.0, 1.0),  # up
    ]
    return scene


def build_pov_view(plotter, turret, board) -> None:
    """Build the turret-POV subplot (call after selecting it)."""
    _add_environment(plotter, board)
    plotter.set_background("#0d141b", top="#1a2b3a")
    update_pov_camera(plotter, turret)


def update_pov_camera(plotter, turret, fov_deg: float = 40.0,
                      aim_distance: float = 50.0) -> None:
    """Point the active subplot's camera along the barrel from the pivot."""
    pivot = turret.base_position + np.array([0.0, 0.0, PIVOT_HEIGHT])
    direction = turret.barrel_direction
    up = (0.0, 0.0, 1.0)
    if abs(direction[2]) > 0.98:  # looking nearly straight up/down
        up = (1.0, 0.0, 0.0)
    plotter.camera.position = tuple(pivot)
    plotter.camera.focal_point = tuple(pivot + direction * aim_distance)
    plotter.camera.up = up
    plotter.camera.view_angle = fov_deg
