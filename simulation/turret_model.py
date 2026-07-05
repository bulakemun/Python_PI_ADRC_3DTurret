"""
Turret plant model: static base, 2-axis (azimuth/elevation) gimbal.

Each axis is modelled as a first-order rate response driven by the
controller's commanded rate: the actual axis rate lags the command through a
time constant ``tau`` (mechanical/servo inertia), and the angle integrates the
actual rate. This makes the PI gains (Kp/Ki) meaningfully affect the closed
loop instead of the plant following commands instantly.

Coordinate/units conventions (see CLAUDE.md):
  - Turret base at ``base_position`` (world origin by default).
  - Azimuth: yaw about +z, measured from the +x forward reference (0 rad =
    facing the target board's nominal bearing).
  - Elevation: pitch from horizontal, positive up.
  - Barrel direction: (cos(el)cos(az), cos(el)sin(az), sin(el)).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class TurretModel:
    """Static-base 2-axis gimbal with first-order rate dynamics per axis."""

    def __init__(
        self,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.01,
        tau: float = 0.2,
        rate_limit: Optional[float] = None,
        elevation_limits: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    ) -> None:
        self.base_position = np.asarray(base_position, dtype=float)
        self.dt = dt
        self.tau = tau
        self.rate_limit = rate_limit
        self.elevation_limits = elevation_limits

        # State: angles (rad) and their actual rates (rad/s).
        self.azimuth = 0.0
        self.elevation = 0.0
        self.azimuth_rate = 0.0
        self.elevation_rate = 0.0

    def reset(self) -> None:
        """Return all state (angles and rates) to zero."""
        self.azimuth = 0.0
        self.elevation = 0.0
        self.azimuth_rate = 0.0
        self.elevation_rate = 0.0

    def _advance_axis(self, angle: float, rate: float, cmd_rate: float):
        """Integrate one axis one timestep with first-order rate lag."""
        if self.rate_limit is not None:
            cmd_rate = float(np.clip(cmd_rate, -self.rate_limit, self.rate_limit))
        # Actual rate relaxes toward the command with time constant tau.
        if self.tau > 0.0:
            rate += (cmd_rate - rate) / self.tau * self.dt
        else:
            rate = cmd_rate
        angle += rate * self.dt
        return angle, rate

    def step(self, azimuth_cmd_speed: float, elevation_cmd_speed: float) -> None:
        """Advance the plant one timestep given commanded axis rates (rad/s)."""
        self.azimuth, self.azimuth_rate = self._advance_axis(
            self.azimuth, self.azimuth_rate, azimuth_cmd_speed
        )
        self.elevation, self.elevation_rate = self._advance_axis(
            self.elevation, self.elevation_rate, elevation_cmd_speed
        )

        lo, hi = self.elevation_limits
        if self.elevation <= lo:
            self.elevation, self.elevation_rate = lo, 0.0
        elif self.elevation >= hi:
            self.elevation, self.elevation_rate = hi, 0.0

    @property
    def orientation(self) -> Tuple[float, float]:
        """Current (azimuth, elevation) in radians."""
        return self.azimuth, self.elevation

    @property
    def barrel_direction(self) -> np.ndarray:
        """Unit vector along the barrel/line of sight in world coordinates."""
        az, el = self.azimuth, self.elevation
        return np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
