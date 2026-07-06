"""
Real-time 3D visualization of the turret and its surroundings, rendered with
PyVista (VTK).

Two renderers share the same simulation state and the same scenery:
  1. World view -- a shaded turret model (static base, a platform that yaws
     with azimuth, and a barrel that pitches with elevation) on a textured
     ground plane, with procedurally-built conifers scattered around and a ring
     of distant mountains (a real elevation model) on the horizon.
  2. Turret POV -- a camera placed at the barrel pivot looking along the barrel,
     so the target board slides toward frame centre as the controller converges.

Scenery assets (ground/bark textures, the mountain mesh) are provided by
``simulation.assets``; anything unavailable falls back to a flat colour. This
module builds the meshes/actors and updates the moving parts each frame. It
owns no control or plant logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv

#: Height of the barrel pivot above the ground (metres).
PIVOT_HEIGHT = 1.2
#: Barrel length (metres).
BARREL_LENGTH = 2.4

# Palette.
_COL_BASE = "#3a3f4b"
_COL_PLATFORM = "#5b6472"
_COL_BARREL = "#8a929e"
_COL_GROUND = "#3d5641"
_COL_TRUNK = "#5b3a21"
_COL_FOLIAGE = "#245c2a"
_COL_BOARD_OUTER = "#c0392b"
_COL_BOARD_MID = "#ecf0f1"
_COL_BOARD_BULL = "#c0392b"

_TEX_CACHE: Dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Transforms.
# --------------------------------------------------------------------------- #
def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _matrix(rot: np.ndarray, translation: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = translation
    return m


def yaw_transform(azimuth: float, pivot: np.ndarray) -> np.ndarray:
    """Transform for parts that yaw with azimuth only (the platform)."""
    return _matrix(_rot_z(azimuth), pivot)


def barrel_transform(azimuth: float, elevation: float, pivot: np.ndarray) -> np.ndarray:
    """Transform mapping the local barrel frame to the world (yaw then pitch)."""
    return _matrix(_rot_z(azimuth) @ _rot_y(-elevation), pivot)


# --------------------------------------------------------------------------- #
# Textures.
# --------------------------------------------------------------------------- #
def _texture(path: Optional[str]):
    """Load (and cache) a texture, repeating at the edges. None on failure."""
    if not path or not os.path.exists(path):
        return None
    if path not in _TEX_CACHE:
        try:
            tex = pv.read_texture(path)
            try:
                tex.repeat = True
            except Exception:
                pass
            _TEX_CACHE[path] = tex
        except Exception:
            _TEX_CACHE[path] = None
    return _TEX_CACHE[path]


# --------------------------------------------------------------------------- #
# Turret meshes.
# --------------------------------------------------------------------------- #
def _turret_platform_mesh() -> pv.PolyData:
    """Yaw platform + trunnion yoke + a commander sight box (local pivot frame)."""
    platform = pv.Cylinder(center=(0, 0, -0.15), direction=(0, 0, 1), radius=0.55,
                           height=0.3)
    yoke_l = pv.Box(bounds=(-0.25, 0.25, 0.45, 0.62, -0.1, 0.55))
    yoke_r = pv.Box(bounds=(-0.25, 0.25, -0.62, -0.45, -0.1, 0.55))
    sight = pv.Box(bounds=(-0.45, -0.2, -0.18, 0.18, 0.15, 0.4))  # rear sight box
    return reduce(lambda a, b: a + b, [platform, yoke_l, yoke_r, sight])


def _turret_barrel_mesh() -> pv.PolyData:
    """Barrel + mantlet + muzzle brake + trunnion (local pivot frame, fires +x)."""
    barrel = pv.Cylinder(center=(BARREL_LENGTH / 2, 0, 0), direction=(1, 0, 0),
                         radius=0.13, height=BARREL_LENGTH)
    mantlet = pv.Box(bounds=(-0.1, 0.5, -0.3, 0.3, -0.3, 0.3))  # gun mantlet
    muzzle = pv.Cylinder(center=(BARREL_LENGTH, 0, 0), direction=(1, 0, 0),
                         radius=0.19, height=0.22)
    breech = pv.Box(bounds=(-0.5, 0.0, -0.24, 0.24, -0.24, 0.24))
    trunnion = pv.Cylinder(center=(0, 0, 0), direction=(0, 1, 0), radius=0.16,
                           height=1.2)
    return reduce(lambda a, b: a + b, [barrel, mantlet, muzzle, breech, trunnion])


@dataclass
class TurretScene:
    """Holds the world-view actors that move each frame and the pivot point."""

    pivot: np.ndarray
    platform_actor: object = field(default=None)
    barrel_actor: object = field(default=None)

    def update(self, turret) -> None:
        az, el = turret.orientation
        if self.platform_actor is not None:
            self.platform_actor.user_matrix = yaw_transform(az, self.pivot)
        if self.barrel_actor is not None:
            self.barrel_actor.user_matrix = barrel_transform(az, el, self.pivot)


# --------------------------------------------------------------------------- #
# Scenery: ground, mountains, trees, board.
# --------------------------------------------------------------------------- #
def _make_mountains(assets: Dict[str, Optional[str]]) -> List[pv.PolyData]:
    """A ring of distant mountains built from the DEM surface, or [] if missing."""
    path = assets.get("mountain")
    if not path or not os.path.exists(path):
        return []
    try:
        base = pv.read(path)
    except Exception:
        return []
    b = base.bounds
    base = base.translate((-(b[0] + b[1]) / 2, -(b[2] + b[3]) / 2, -b[4]),
                          inplace=False)
    b = base.bounds
    width = max(b[1] - b[0], b[3] - b[2], 1.0)

    meshes = []
    n = 7
    rng = np.random.default_rng(3)
    for i in range(n):
        ang = 2.0 * np.pi * i / n + rng.uniform(-0.15, 0.15)
        target_w = rng.uniform(480.0, 760.0)
        radius = rng.uniform(680.0, 900.0)
        m = base.copy()
        m.points *= target_w / width
        m["elev"] = m.points[:, 2]
        m.rotate_z(rng.uniform(0, 360), inplace=True)
        m.translate((radius * np.cos(ang), radius * np.sin(ang), -30.0), inplace=True)
        meshes.append(m)
    return meshes


def _make_trees(assets: Dict[str, Optional[str]], count: int = 26, seed: int = 7):
    """Return (trunk_mesh, foliage_mesh, bark_texture) for scattered conifers."""
    rng = np.random.default_rng(seed)
    trunks, foliage = [], []
    for _ in range(count):
        r = rng.uniform(7.0, 65.0)
        a = rng.uniform(0.0, 2.0 * np.pi)
        x, y = r * np.cos(a), r * np.sin(a)
        h = rng.uniform(2.6, 4.8)
        trunk_h = h * 0.30
        trunks.append(pv.Cylinder(center=(x, y, trunk_h / 2), direction=(0, 0, 1),
                                  radius=h * 0.05, height=trunk_h, resolution=10))
        layers = 3
        for k in range(layers):
            cone_h = h * 0.55 * (1.0 - 0.16 * k)
            cone_r = h * (0.30 - 0.07 * k)
            cz = trunk_h + (h - trunk_h) * (k / layers) * 0.9 + cone_h * 0.2
            foliage.append(pv.Cone(center=(x, y, cz), direction=(0, 0, 1),
                                   height=cone_h, radius=cone_r, resolution=9))
    trunk_mesh = reduce(lambda p, q: p.merge(q), trunks)
    foliage_mesh = reduce(lambda p, q: p.merge(q), foliage)
    return trunk_mesh, foliage_mesh, _texture(assets.get("bark"))


def _board_meshes(board):
    """Bullseye rings for the target board, facing back toward the turret (-x)."""
    center = board.position
    normal = (1.0, 0.0, 0.0)
    r = min(board.width, board.height_dim) / 2.0
    outer = pv.Disc(center=center - np.array([0.05, 0, 0]), inner=0.0, outer=r,
                    normal=normal, c_res=48)
    mid = pv.Disc(center=center - np.array([0.10, 0, 0]), inner=0.0, outer=r * 0.62,
                  normal=normal, c_res=48)
    bull = pv.Disc(center=center - np.array([0.15, 0, 0]), inner=0.0, outer=r * 0.25,
                   normal=normal, c_res=48)
    return [(outer, _COL_BOARD_OUTER), (mid, _COL_BOARD_MID), (bull, _COL_BOARD_BULL)]


def build_environment(assets: Dict[str, Optional[str]]) -> dict:
    """Pre-build shared scenery so both subplots render identical surroundings."""
    return {
        "assets": assets,
        "mountains": _make_mountains(assets),
        "trees": _make_trees(assets),
    }


def _add_ground(plotter, assets, size: float = 1800.0) -> None:
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=size, j_size=size,
                     i_resolution=1, j_resolution=1)
    tex = _texture(assets.get("grass"))
    if tex is not None:
        tc = plane.active_texture_coordinates
        if tc is not None:
            plane.active_texture_coordinates = tc * (size / 12.0)  # tile ~12 m
        plotter.add_mesh(plane, texture=tex, ambient=0.28, diffuse=0.9)
    else:
        plotter.add_mesh(plane, color=_COL_GROUND, ambient=0.3)


def _add_environment(plotter, board, env: dict) -> None:
    """Add ground, mountains, trees and the board to the active subplot."""
    _add_ground(plotter, env["assets"])

    for m in env["mountains"]:
        plotter.add_mesh(m, scalars="elev", cmap="gist_earth", show_scalar_bar=False,
                         ambient=0.25, smooth_shading=True)

    trunk_mesh, foliage_mesh, bark = env["trees"]
    if bark is not None:
        plotter.add_mesh(trunk_mesh.texture_map_to_plane(), texture=bark, ambient=0.3)
    else:
        plotter.add_mesh(trunk_mesh, color=_COL_TRUNK, ambient=0.3)
    plotter.add_mesh(foliage_mesh, color=_COL_FOLIAGE, ambient=0.28,
                     smooth_shading=True)

    for mesh, color in _board_meshes(board):
        plotter.add_mesh(mesh, color=color, ambient=0.35, smooth_shading=True)


# --------------------------------------------------------------------------- #
# View builders.
# --------------------------------------------------------------------------- #
def build_world_view(plotter, turret, board, env: dict) -> TurretScene:
    """Build the world-view subplot (call after selecting it). Returns a scene."""
    pivot = turret.base_position + np.array([0.0, 0.0, PIVOT_HEIGHT])

    _add_environment(plotter, board, env)

    base = pv.Cylinder(
        center=turret.base_position + np.array([0, 0, PIVOT_HEIGHT / 2]),
        direction=(0, 0, 1), radius=0.5, height=PIVOT_HEIGHT,
    )
    plotter.add_mesh(base, color=_COL_BASE, ambient=0.25, smooth_shading=True)
    platform_actor = plotter.add_mesh(_turret_platform_mesh(), color=_COL_PLATFORM,
                                      ambient=0.25, smooth_shading=True)
    barrel_actor = plotter.add_mesh(_turret_barrel_mesh(), color=_COL_BARREL,
                                    metallic=0.4, roughness=0.5, smooth_shading=True)

    scene = TurretScene(pivot=pivot, platform_actor=platform_actor,
                        barrel_actor=barrel_actor)
    scene.update(turret)

    plotter.add_axes()
    plotter.set_background("#20344a", top="#7ea6c9")
    plotter.camera_position = [
        (10.0, -11.0, 6.5),
        (0.0, 0.0, PIVOT_HEIGHT),
        (0.0, 0.0, 1.0),
    ]
    return scene


def build_pov_view(plotter, turret, board, env: dict) -> None:
    """Build the turret-POV subplot (call after selecting it)."""
    _add_environment(plotter, board, env)
    plotter.set_background("#16283b", top="#6f97bd")
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
