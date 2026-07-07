"""
Stewart-platform base disturbance.

The turret is mounted on a Stewart (hexapod) platform that is *not* held still:
it injects sinusoidal **yaw** and **pitch** disturbances at the base, so the
turret's line of sight is perturbed and the control loop has to stabilise
against it. This models a gyro-stabilised turret on a moving mount.

The disturbance is a pure kinematic angle source (no plant dynamics of its own):
each axis is ``magnitude * sin(2*pi*frequency*t)`` with a quarter-cycle phase
offset between yaw and pitch so they don't move in lockstep. Magnitudes are held
in degrees so they map directly onto the GUI sliders (3-15 deg), frequencies in
Hz (0.1-0.4 Hz).
"""

from __future__ import annotations

import numpy as np


class StewartDisturbance:
    """Sinusoidal yaw/pitch base disturbance for the turret mount."""

    def __init__(
        self,
        yaw_mag_deg: float = 6.0,
        yaw_freq_hz: float = 0.2,
        pitch_mag_deg: float = 5.0,
        pitch_freq_hz: float = 0.25,
        pitch_phase: float = np.pi / 2.0,
        enabled: bool = True,
        ramp_time: float = 0.3,
    ) -> None:
        # Kept in degrees / Hz so the GUI sliders write straight to these.
        self.yaw_mag_deg = yaw_mag_deg
        self.yaw_freq_hz = yaw_freq_hz
        self.pitch_mag_deg = pitch_mag_deg
        self.pitch_freq_hz = pitch_freq_hz
        self.pitch_phase = pitch_phase
        # Toggling ``enabled`` ramps an envelope in/out over ``ramp_time`` seconds
        # so the platform starts/stops smoothly instead of jumping the base angle
        # (which would jerk the turret). Slider magnitudes are preserved.
        self.enabled = enabled
        self.ramp_time = ramp_time
        self._env = 1.0 if enabled else 0.0

    def advance(self, dt: float) -> None:
        """Advance the on/off envelope toward the current ``enabled`` state."""
        target = 1.0 if self.enabled else 0.0
        step = dt / self.ramp_time if self.ramp_time > 0 else 1.0
        if self._env < target:
            self._env = min(target, self._env + step)
        elif self._env > target:
            self._env = max(target, self._env - step)

    def yaw(self, t: float) -> float:
        """Base yaw disturbance at time ``t`` (radians), scaled by the envelope."""
        return self._env * np.radians(self.yaw_mag_deg) * np.sin(
            2.0 * np.pi * self.yaw_freq_hz * t
        )

    def pitch(self, t: float) -> float:
        """Base pitch disturbance at time ``t`` (radians), scaled by the envelope."""
        return self._env * np.radians(self.pitch_mag_deg) * np.sin(
            2.0 * np.pi * self.pitch_freq_hz * t + self.pitch_phase
        )

    def angles(self, t: float):
        """Convenience: ``(yaw, pitch)`` base disturbance at ``t`` (radians)."""
        return self.yaw(t), self.pitch(t)
