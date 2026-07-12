"""
Turret plant models: static base, 2-axis (azimuth/elevation) gimbal.

Two selectable plants share one public interface (``step(az_cmd, el_cmd)``,
``azimuth``/``elevation``, ``*_rate``, ``orientation``, ``barrel_direction``,
``reset``):

  - ``TurretModel`` -- the original **kinematic velocity servo**: the controller
    commands a rate, the actual rate lags it through a time constant ``tau``, and
    the angle integrates the rate. Torque and inertia are lumped into ``tau``.
  - ``TorqueTurretModel`` -- a **torque-driven second-order** axis with real
    inertia ``J``, viscous damping ``B``, motor torque constant ``K_t`` +
    current limit ``i_max`` (torque saturation), and basic Coulomb friction with
    a Karnopp stiction band. Here the controller commands a **current** ``i_cmd``
    and the plant makes ``tau_cmd = K_t * i_cmd``.

Coordinate/units conventions (see CLAUDE.md):
  - Turret base at ``base_position`` (world origin by default).
  - Azimuth: yaw about +z, from the +x forward reference (0 = board bearing).
  - Elevation: pitch from horizontal, positive up.
  - Barrel direction: (cos(el)cos(az), cos(el)sin(az), sin(el)).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class GimbalPlant:
    """Shared 2-axis gimbal state + geometry (base for the plant models)."""

    def __init__(self, base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.base_position = np.asarray(base_position, dtype=float)
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


class TurretModel(GimbalPlant):
    """Static-base 2-axis gimbal with first-order rate dynamics per axis.

    The controller commands a **rate** (rad/s); the actual rate relaxes toward it
    with time constant ``tau`` and the angle integrates the actual rate.
    """

    def __init__(
        self,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.01,
        tau: float = 0.2,
        rate_limit: Optional[float] = None,
        elevation_limits: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
    ) -> None:
        super().__init__(base_position)
        self.dt = dt
        self.tau = tau
        self.rate_limit = rate_limit
        self.elevation_limits = elevation_limits

    def _advance_axis(self, angle: float, rate: float, cmd_rate: float):
        """Integrate one axis one timestep with first-order rate lag."""
        if self.rate_limit is not None:
            cmd_rate = float(np.clip(cmd_rate, -self.rate_limit, self.rate_limit))
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
        _clamp_elevation(self)


def _pair(v) -> Tuple[float, float]:
    """Expand a scalar to ``(v, v)`` or pass a 2-sequence through (per-axis)."""
    if np.isscalar(v):
        return float(v), float(v)
    return float(v[0]), float(v[1])


class TorqueTurretModel(GimbalPlant):
    """Torque-driven second-order gimbal per axis.

    Per axis (semi-implicit Euler, ``dt``):

        tau_m  = clip(K_t * i_cmd, -tau_max, +tau_max)   # torque saturation
        tau_fr = B * w + tau_c * sgn(w)                  # viscous + Coulomb
        w     += (tau_m - tau_fr) / J * dt
        phi   += w * dt

    Coulomb friction uses a Karnopp stiction band to avoid chatter at ``w ~= 0``:
    if ``|w| < w_eps`` and the net applied torque ``|tau_m - B*w| <= tau_c`` the
    axis is held (``w = 0``, friction absorbs the torque); otherwise the axis
    breaks away and Coulomb friction opposes the motion.

    ``J`` differs per axis (azimuth carries the whole gimbal, so ``J_az > J_el``).
    ``B``, ``tau_c``, ``K_t``, ``i_max`` default shared but accept per-axis
    ``(az, el)`` tuples. ``tau_max = K_t * i_max``.
    """

    def __init__(
        self,
        base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.01,
        J_az: float = 6.0,
        J_el: float = 2.5,
        B: float = 1.2,
        tau_c: float = 0.8,
        K_t: float = 0.8,
        i_max: float = 50.0,
        rate_limit: Optional[float] = None,
        elevation_limits: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        w_eps: float = 1e-3,
    ) -> None:
        super().__init__(base_position)
        self.dt = dt
        self.rate_limit = rate_limit           # drive speed limit (rad/s)
        self.elevation_limits = elevation_limits
        self.w_eps = w_eps                     # Karnopp stiction velocity band

        self.J_az, self.J_el = float(J_az), float(J_el)
        self.B_az, self.B_el = _pair(B)
        self.tau_c_az, self.tau_c_el = _pair(tau_c)
        self.K_t_az, self.K_t_el = _pair(K_t)
        self.i_max_az, self.i_max_el = _pair(i_max)

    @property
    def tau_max_az(self) -> float:
        return self.K_t_az * self.i_max_az

    @property
    def tau_max_el(self) -> float:
        return self.K_t_el * self.i_max_el

    def _advance_axis(self, angle, omega, i_cmd, J, B, tau_c, K_t, i_max):
        """Integrate one torque-driven axis one timestep (Karnopp friction)."""
        dt = self.dt
        tau_max = K_t * i_max
        tau_m = float(np.clip(K_t * i_cmd, -tau_max, tau_max))

        if abs(omega) < self.w_eps and abs(tau_m - B * omega) <= tau_c:
            omega = 0.0                        # static friction holds it (no creep)
        else:
            # Coulomb opposes motion; at breakaway it opposes the net applied torque.
            fr_dir = np.sign(omega) if abs(omega) >= self.w_eps \
                else np.sign(tau_m - B * omega)
            tau_fr = B * omega + tau_c * fr_dir
            omega += (tau_m - tau_fr) / J * dt

        if self.rate_limit is not None:
            omega = float(np.clip(omega, -self.rate_limit, self.rate_limit))
        angle += omega * dt
        return angle, omega

    def step(self, azimuth_cmd_current: float, elevation_cmd_current: float) -> None:
        """Advance the plant one timestep given commanded currents (A)."""
        self.azimuth, self.azimuth_rate = self._advance_axis(
            self.azimuth, self.azimuth_rate, azimuth_cmd_current,
            self.J_az, self.B_az, self.tau_c_az, self.K_t_az, self.i_max_az,
        )
        self.elevation, self.elevation_rate = self._advance_axis(
            self.elevation, self.elevation_rate, elevation_cmd_current,
            self.J_el, self.B_el, self.tau_c_el, self.K_t_el, self.i_max_el,
        )
        _clamp_elevation(self)


def _clamp_elevation(plant) -> None:
    """Apply the elevation hard stop (rate zeroed at the stop)."""
    lo, hi = plant.elevation_limits
    if plant.elevation <= lo:
        plant.elevation, plant.elevation_rate = lo, 0.0
    elif plant.elevation >= hi:
        plant.elevation, plant.elevation_rate = hi, 0.0
