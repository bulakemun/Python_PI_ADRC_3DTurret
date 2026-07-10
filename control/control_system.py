"""
Cascade control system for the turret, with selectable control modes.

Architecture (per axis) -- an outer position loop wrapped around the existing
inner speed loop:

    position ref --(+)-->[ Kp_pos ]--> speed ref --(+)-->[ PI: Kp_s, Ki_s ]--> rate cmd
                   -                                  -
                 LOS angle                         axis rate

  * Inner speed loop (PI, Kp_s/Ki_s): regulates the axis angular *rate* to a
    speed reference by commanding the plant (a velocity servo).
  * Outer position loop (P, Kp_pos in 1..20 [1/s]): turns a position error into
    the inner loop's speed reference.

Modes:
  1. SPEED    -- the reference signal is a *speed* reference (rad/s); only the
                 inner speed loop runs. "Like we do now", but rate-commanded.
  2. POSITION -- the reference signal is a *position* reference (rad); the full
                 cascade runs. A constant reference is available (a DC setpoint).
  3. TARGET   -- the position reference is the geometric target angle, so the
                 loop drives the barrel/target position error to zero (auto-aim).

Units: everything here is SI (radians, rad/s, seconds). The GUI is responsible
for converting slider values in degrees / deg/s / Hz to these units; see
``ControlSystem.reference`` which reads ``amplitude_rad`` already in radians.
All controllers regulate the *line of sight* (base disturbance + gimbal), so the
loop stabilises pointing against the Stewart-platform base motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np

from control.pi_controller import PIController
from control.reference_signals import square_wave, sine_wave, constant_wave


class Mode(IntEnum):
    SPEED = 1
    POSITION = 2
    TARGET = 3


@dataclass
class AxisResult:
    """Per-axis outcome of one control step."""

    rate_cmd: float   # commanded axis rate to the plant (rad/s)
    error: float      # loop error used for the graph (rad in pos/target, rad/s in speed)


class ControlSystem:
    """Two-axis cascade controller with SPEED / POSITION / TARGET modes.

    Gains and reference settings are plain attributes so the GUI can mutate them
    live. Call :meth:`step` once per plant timestep.
    """

    def __init__(self, dt: float, rate_limit: float, el_hold: float = 0.0) -> None:
        self.dt = dt
        self.rate_limit = rate_limit          # plant rate command limit (rad/s)
        self.max_speed_ref = rate_limit       # outer-loop speed-ref clamp (rad/s)
        self.el_hold = el_hold                # elevation setpoint pointing at board (rad)

        # Live-tunable settings.
        self.mode = Mode.SPEED
        self.kp_pos = 8.0                     # outer position P gain [1/s]
        self.kp_speed = 6.0                   # inner speed PI
        self.ki_speed = 2.0
        self.signal = "square"                # square | sine | constant
        self.amplitude_rad = np.radians(15.0)  # deg or deg/s already -> radians
        self.frequency = 0.3                  # Hz
        # SPEED mode still commands via the speed loop, but this optionally
        # swaps the *reported* error (graph/CSV) for the target-tracking
        # position error, for a quick look at aim accuracy without leaving
        # SPEED mode.
        self.speed_shows_target_error = False

        limits = (-rate_limit, rate_limit)
        speed_limits = (-self.max_speed_ref, self.max_speed_ref)
        # Outer position P controllers (Ki=0) output a speed reference.
        self._az_pos = PIController(self.kp_pos, 0.0, dt, output_limits=speed_limits)
        self._el_pos = PIController(self.kp_pos, 0.0, dt, output_limits=speed_limits)
        # Inner speed PI controllers output a rate command to the plant.
        self._az_spd = PIController(self.kp_speed, self.ki_speed, dt, output_limits=limits)
        self._el_spd = PIController(self.kp_speed, self.ki_speed, dt, output_limits=limits)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear all integrators (call when switching modes)."""
        for c in (self._az_pos, self._el_pos, self._az_spd, self._el_spd):
            c.reset()

    def reference(self, t: float) -> float:
        """Current reference value (radians or rad/s) for the active signal."""
        a, f = self.amplitude_rad, self.frequency
        if self.signal == "sine":
            return float(sine_wave(t, a, f))
        if self.signal == "constant":
            return float(constant_wave(t, a, f))
        return float(square_wave(t, a, f))

    @property
    def error_is_rate(self) -> bool:
        """True when the graphed error is a rate (SPEED mode) vs a position."""
        return self.mode == Mode.SPEED and not self.speed_shows_target_error

    def _sync_gains(self) -> None:
        self._az_pos.kp = self._el_pos.kp = self.kp_pos
        self._az_spd.kp = self._el_spd.kp = self.kp_speed
        self._az_spd.ki = self._el_spd.ki = self.ki_speed

    # ------------------------------------------------------------------ #
    def _speed_loop(self, ctrl: PIController, speed_ref: float, rate_meas: float
                    ) -> float:
        """Inner loop: command a plant rate to make the axis rate track speed_ref."""
        return ctrl.step(speed_ref - rate_meas)

    def step(
        self,
        t: float,
        los_az: float, rate_az: float,
        los_el: float, rate_el: float,
        target_az: float, target_el: float,
    ) -> Tuple[AxisResult, AxisResult]:
        """Advance the controller one timestep.

        Args (all SI, radians / rad/s):
            t: current time.
            los_az/los_el: measured line-of-sight angles (base + gimbal).
            rate_az/rate_el: measured line-of-sight rates (gyro: gimbal + base
                rate), so the inner speed loop stabilises the absolute pointing.
            target_az/target_el: geometric target angles (for TARGET mode; the
                elevation also serves as the POSITION-mode elevation setpoint).

        Returns:
            (azimuth, elevation) :class:`AxisResult`.
        """
        self._sync_gains()

        if self.mode == Mode.SPEED:
            # Reference is a speed (rad/s); elevation is commanded to hold still.
            az_speed_ref = self.reference(t)
            el_speed_ref = 0.0
            az_err = az_speed_ref - rate_az
            el_err = el_speed_ref - rate_el
            az_cmd = self._az_spd.step(az_err)
            el_cmd = self._el_spd.step(el_err)
            if self.speed_shows_target_error:
                az_pos_err = target_az - los_az
                el_pos_err = target_el - los_el
                return (AxisResult(az_cmd, az_pos_err), AxisResult(el_cmd, el_pos_err))
            return (AxisResult(az_cmd, az_err), AxisResult(el_cmd, el_err))

        # POSITION / TARGET: full cascade on the line-of-sight position.
        if self.mode == Mode.TARGET:
            az_ref, el_ref = target_az, target_el
        else:  # POSITION
            az_ref, el_ref = self.reference(t), self.el_hold

        az_pos_err = az_ref - los_az
        el_pos_err = el_ref - los_el
        az_speed_ref = self._az_pos.step(az_pos_err)   # P -> speed ref (rad/s)
        el_speed_ref = self._el_pos.step(el_pos_err)
        az_cmd = self._speed_loop(self._az_spd, az_speed_ref, rate_az)
        el_cmd = self._speed_loop(self._el_spd, el_speed_ref, rate_el)
        return (AxisResult(az_cmd, az_pos_err), AxisResult(el_cmd, el_pos_err))
