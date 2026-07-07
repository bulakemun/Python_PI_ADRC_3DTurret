"""
Reference signal generators for the control loop: square wave and sine wave.

These are physics-agnostic: they operate on time values (scalar or numpy
array) and return the commanded reference in the same shape. Amplitude,
frequency and offset are exposed as adaptive sliders in the PyVista UI
(see app.py). Units are whatever the caller uses for amplitude/offset
(the app drives these in radians after converting from slider degrees).
"""

from __future__ import annotations

import numpy as np


def square_wave(t, amplitude=1.0, frequency=1.0, duty=0.5, offset=0.0):
    """Bipolar square wave oscillating between ``offset ± amplitude``.

    Args:
        t: Time (seconds), scalar or array-like.
        amplitude: Peak deviation from ``offset``.
        frequency: Cycles per second (Hz).
        duty: Fraction of each period spent at the high level, in [0, 1].
        offset: Constant added to the whole signal (the nominal setpoint).

    Returns:
        The reference value(s), matching the shape of ``t``.
    """
    t = np.asarray(t, dtype=float)
    phase = np.mod(t * frequency, 1.0)
    return np.where(phase < duty, amplitude, -amplitude) + offset


def sine_wave(t, amplitude=1.0, frequency=1.0, phase=0.0, offset=0.0):
    """Sine wave oscillating between ``offset ± amplitude``.

    Args:
        t: Time (seconds), scalar or array-like.
        amplitude: Peak deviation from ``offset``.
        frequency: Cycles per second (Hz).
        phase: Phase shift in radians.
        offset: Constant added to the whole signal (the nominal setpoint).

    Returns:
        The reference value(s), matching the shape of ``t``.
    """
    t = np.asarray(t, dtype=float)
    return amplitude * np.sin(2.0 * np.pi * frequency * t + phase) + offset


def constant_wave(t, amplitude=1.0, frequency=1.0, offset=0.0):
    """Constant reference at ``amplitude + offset`` (a step / DC setpoint).

    ``frequency`` is accepted and ignored so this shares the generators'
    signature and can be selected interchangeably in the UI.
    """
    t = np.asarray(t, dtype=float)
    return np.full(t.shape, amplitude + offset) if t.ndim else amplitude + offset


#: Registry so the UI can offer signals by name and stay open to extension.
REFERENCE_SIGNALS = {
    "square": square_wave,
    "sine": sine_wave,
    "constant": constant_wave,
}
