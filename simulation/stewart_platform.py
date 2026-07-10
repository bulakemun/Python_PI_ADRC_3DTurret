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
        self._env_rate = 0.0   # d(env)/dt this step, for the gyro rate
        # Per-axis phase references. Zero by default (waveforms are the plain
        # sin(ωt) / sin(ωt+φ)); ``restart`` moves them to a zero crossing.
        self._yaw_t0 = 0.0
        self._pitch_t0 = 0.0

    def advance(self, dt: float) -> None:
        """Advance the on/off envelope toward the current ``enabled`` state.

        Records the exact envelope slope ``_env_rate`` so the gyro rate can
        include the ``ė·A·sin`` term while the envelope is ramping.
        """
        target = 1.0 if self.enabled else 0.0
        step = dt / self.ramp_time if self.ramp_time > 0 else 1.0
        prev = self._env
        if self._env < target:
            self._env = min(target, self._env + step)
        elif self._env > target:
            self._env = max(target, self._env - step)
        self._env_rate = (self._env - prev) / dt if dt > 0 else 0.0

    def snap(self, env: float) -> None:
        """Force the envelope to ``env`` immediately (no ramp), e.g. on a fold."""
        self._env = env
        self._env_rate = 0.0

    def restart(self, t: float) -> None:
        """Restart both sinusoids at a zero crossing at time ``t``.

        Enabling the platform mid-cycle would otherwise sweep the base angle from
        0 up to ``A·sin(ωt)`` while the envelope ramps — a *position* injection
        that a pure rate loop (SPEED mode, no position feedback) cannot undo,
        leaving a permanent line-of-sight offset. Starting each axis from a zero
        crossing means the base grows smoothly from rest, as a real platform must.
        """
        self._yaw_t0 = t
        w_p = 2.0 * np.pi * self.pitch_freq_hz
        # Offset so the pitch argument (which carries ``pitch_phase``) is 0 at t.
        self._pitch_t0 = t + (self.pitch_phase / w_p if w_p else 0.0)

    def _yaw_arg(self, t: float) -> float:
        return 2.0 * np.pi * self.yaw_freq_hz * (t - self._yaw_t0)

    def _pitch_arg(self, t: float) -> float:
        return (2.0 * np.pi * self.pitch_freq_hz * (t - self._pitch_t0)
                + self.pitch_phase)

    def yaw(self, t: float) -> float:
        """Base yaw disturbance at time ``t`` (radians), scaled by the envelope."""
        return self._env * np.radians(self.yaw_mag_deg) * np.sin(self._yaw_arg(t))

    def pitch(self, t: float) -> float:
        """Base pitch disturbance at time ``t`` (radians), scaled by the envelope."""
        return self._env * np.radians(self.pitch_mag_deg) * np.sin(self._pitch_arg(t))

    def yaw_rate(self, t: float) -> float:
        """Base yaw angular rate at ``t`` (rad/s) -- what a gyro would sense.

        Full derivative of ``β = e·A·sin(θ)``. The ``ė·A·sin`` term matters while
        the on/off envelope ramps: without it the loop never sees the base angle
        sweeping in.
        """
        w = 2.0 * np.pi * self.yaw_freq_hz
        a = np.radians(self.yaw_mag_deg)
        th = self._yaw_arg(t)
        return self._env_rate * a * np.sin(th) + self._env * a * w * np.cos(th)

    def pitch_rate(self, t: float) -> float:
        """Base pitch angular rate at ``t`` (rad/s) -- what a gyro would sense."""
        w = 2.0 * np.pi * self.pitch_freq_hz
        a = np.radians(self.pitch_mag_deg)
        th = self._pitch_arg(t)
        return self._env_rate * a * np.sin(th) + self._env * a * w * np.cos(th)

    def angles(self, t: float):
        """Convenience: ``(yaw, pitch)`` base disturbance at ``t`` (radians)."""
        return self.yaw(t), self.pitch(t)

    def rates(self, t: float):
        """Convenience: ``(yaw_rate, pitch_rate)`` base rate at ``t`` (rad/s)."""
        return self.yaw_rate(t), self.pitch_rate(t)
