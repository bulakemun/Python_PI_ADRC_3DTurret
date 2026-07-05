"""
PI speed controller for the turret azimuth/elevation axes.

Discrete-time PI controller:

    u[k] = Kp * e[k] + Ki * integral,   integral += e[k] * dt

The controller is physics-agnostic: it consumes a scalar error and returns a
scalar command (here, a commanded axis rate). Output saturation with
back-calculation anti-windup keeps the integral from running away while the
command is clamped. One instance is used per axis.
"""

from __future__ import annotations

from typing import Optional, Tuple


class PIController:
    """Discrete-time PI controller with output limits and anti-windup."""

    def __init__(
        self,
        kp: float,
        ki: float,
        dt: float,
        output_limits: Tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.output_limits = output_limits
        self._integral = 0.0

    def reset(self) -> None:
        """Clear the integrator (call before starting a new run)."""
        self._integral = 0.0

    @property
    def integral(self) -> float:
        return self._integral

    def step(self, error: float) -> float:
        """Advance one timestep and return the (possibly saturated) command."""
        self._integral += error * self.dt
        u = self.kp * error + self.ki * self._integral

        lo, hi = self.output_limits
        u_clamped = u
        if hi is not None and u_clamped > hi:
            u_clamped = hi
        if lo is not None and u_clamped < lo:
            u_clamped = lo

        # Anti-windup: if the output saturated, back-calculate the integral so
        # that ki*integral accounts for exactly the clamped headroom. This
        # stops the integrator winding up while the command is pinned.
        if u_clamped != u and self.ki != 0.0:
            self._integral = (u_clamped - self.kp * error) / self.ki

        return u_clamped
