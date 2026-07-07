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
    ) -> None:
        # Kept in degrees / Hz so the GUI sliders write straight to these.
        self.yaw_mag_deg = yaw_mag_deg
        self.yaw_freq_hz = yaw_freq_hz
        self.pitch_mag_deg = pitch_mag_deg
        self.pitch_freq_hz = pitch_freq_hz
        self.pitch_phase = pitch_phase
        # When disabled the platform holds still; slider magnitudes are preserved.
        self.enabled = enabled

    def yaw(self, t: float) -> float:
        """Base yaw disturbance at time ``t`` (radians)."""
        if not self.enabled:
            return 0.0
        return np.radians(self.yaw_mag_deg) * np.sin(2.0 * np.pi * self.yaw_freq_hz * t)

    def pitch(self, t: float) -> float:
        """Base pitch disturbance at time ``t`` (radians)."""
        if not self.enabled:
            return 0.0
        return np.radians(self.pitch_mag_deg) * np.sin(
            2.0 * np.pi * self.pitch_freq_hz * t + self.pitch_phase
        )

    def angles(self, t: float):
        """Convenience: ``(yaw, pitch)`` base disturbance at ``t`` (radians)."""
        return self.yaw(t), self.pitch(t)
