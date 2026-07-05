"""
Static target board, nominally 400 m downrange along the turret's forward
reference direction.

The board is a flat rectangle whose face is perpendicular to the turret's
forward (+x) axis, centred at ``(distance, 0, height)``. It provides the
line-of-sight geometry the visualization needs and the nominal azimuth/
elevation the turret must reach to point at it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

TARGET_DISTANCE_M = 400.0


class TargetBoard:
    """Static rectangular target board and its line-of-sight geometry."""

    def __init__(
        self,
        distance_m: float = TARGET_DISTANCE_M,
        height_m: float = 0.0,
        width: float = 4.0,
        height_dim: float = 4.0,
        bearing: float = 0.0,
    ) -> None:
        self.distance_m = distance_m
        self.height_m = height_m
        self.width = width  # horizontal extent (along world y), metres
        self.height_dim = height_dim  # vertical extent (along world z), metres
        self.bearing = bearing  # nominal azimuth of the board centre (rad)

    @property
    def position(self) -> np.ndarray:
        """Board centre ``(x, y, z)`` in world coordinates (metres)."""
        return np.array(
            [
                self.distance_m * np.cos(self.bearing),
                self.distance_m * np.sin(self.bearing),
                self.height_m,
            ]
        )

    def corners(self) -> np.ndarray:
        """Return the four board corners (4x3) for rendering the face.

        The face spans ``width`` horizontally (world y) and ``height_dim``
        vertically (world z), so it is presented flat toward the turret.
        Ordered counter-clockwise for a clean polygon.
        """
        cx, cy, cz = self.position
        hw, hh = self.width / 2.0, self.height_dim / 2.0
        return np.array(
            [
                [cx, cy - hw, cz - hh],
                [cx, cy + hw, cz - hh],
                [cx, cy + hw, cz + hh],
                [cx, cy - hw, cz + hh],
            ]
        )

    def required_angles(
        self, from_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Tuple[float, float]:
        """Azimuth/elevation (rad) to point at the board centre from a point."""
        d = self.position - np.asarray(from_position, dtype=float)
        azimuth = np.arctan2(d[1], d[0])
        elevation = np.arctan2(d[2], np.hypot(d[0], d[1]))
        return float(azimuth), float(elevation)
