"""
Entry point for the 3D Turret Simulation (PyVista / VTK).

Opens an interactive window with two live 3D views -- a world view of the
turret model and a turret-POV camera -- and in-window sliders to tune the
control loop while it runs. This is the only module that wires ``control`` and
``simulation`` together; it keeps the logic thin.

Per-axis loop (see CLAUDE.md):  error = reference - angle -> PI -> commanded
rate -> the turret integrates the rate. The reference signal drives azimuth;
elevation holds the angle that points at the board centre.

Run with:
    uv run python app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from control.pi_controller import PIController
from control.reference_signals import square_wave, sine_wave
from simulation.turret_model import TurretModel
from simulation.target_board import TargetBoard
from simulation import visualization as viz

# Wall-clock pacing: advance this many plant steps per rendered frame so the
# animation runs at roughly real time (FRAME_INTERVAL_MS * SUBSTEPS ~= dt*1000).
DT = 0.01
SUBSTEPS = 3
FRAME_INTERVAL_MS = 30
RATE_LIMIT = np.radians(90.0)


@dataclass
class LoopParams:
    """Live-tunable parameters, mutated by the slider/checkbox callbacks."""

    kp: float = 6.0
    ki: float = 2.0
    amplitude: float = np.radians(15.0)  # azimuth reference amplitude (rad)
    frequency: float = 0.3               # Hz
    use_sine: bool = False               # False -> square wave


@dataclass
class SimState:
    """Everything that persists across animation frames."""

    turret: TurretModel
    board: TargetBoard
    az_ctrl: PIController
    el_ctrl: PIController
    params: LoopParams
    el_setpoint: float
    t: float = 0.0
    az_ref: float = 0.0
    az_err: float = 0.0


def _reference(params: LoopParams, t: float) -> float:
    """Current azimuth reference (rad) for the selected signal."""
    if params.use_sine:
        return float(sine_wave(t, params.amplitude, params.frequency))
    return float(square_wave(t, params.amplitude, params.frequency))


def _advance(state: SimState) -> None:
    """Step the closed loop by SUBSTEPS plant steps."""
    p = state.params
    state.az_ctrl.kp = state.el_ctrl.kp = p.kp
    state.az_ctrl.ki = state.el_ctrl.ki = p.ki
    for _ in range(SUBSTEPS):
        state.az_ref = _reference(p, state.t)
        az_err = state.az_ref - state.turret.azimuth
        el_err = state.el_setpoint - state.turret.elevation
        state.turret.step(state.az_ctrl.step(az_err), state.el_ctrl.step(el_err))
        state.t += DT
    state.az_err = az_err


def _build_state() -> SimState:
    board = TargetBoard(height_m=14.0, width=30.0, height_dim=24.0)
    turret = TurretModel(dt=DT, tau=0.2, rate_limit=RATE_LIMIT)
    params = LoopParams()
    limits = (-RATE_LIMIT, RATE_LIMIT)
    _, el_setpoint = board.required_angles(turret.base_position)
    return SimState(
        turret=turret,
        board=board,
        az_ctrl=PIController(params.kp, params.ki, DT, output_limits=limits),
        el_ctrl=PIController(params.kp, params.ki, DT, output_limits=limits),
        params=params,
        el_setpoint=el_setpoint,
    )


def build_plotter(off_screen: bool = False):
    """Assemble the two-view plotter, widgets, and animation tick.

    Returns ``(plotter, state, tick)``. ``tick`` advances the simulation and
    refreshes both views; ``main`` drives it on a timer, and the headless test
    can call it directly.
    """
    state = _build_state()
    pl = pv.Plotter(shape=(1, 2), window_size=(1500, 760), off_screen=off_screen)
    pl.subplot(0, 0)
    scene = viz.build_world_view(pl, state.turret, state.board)
    pl.add_text("World view", position="upper_edge", font_size=11, color="white")

    pl.subplot(0, 1)
    viz.build_pov_view(pl, state.turret, state.board)
    pl.add_text("Turret POV", position="upper_edge", font_size=11, color="white")

    def tick(*_args) -> None:
        _advance(state)
        pl.subplot(0, 0)
        scene.update(state.turret)
        pl.add_text(
            f"Kp={state.params.kp:.1f}  Ki={state.params.ki:.1f}   "
            f"az err={np.degrees(state.az_err):+5.1f} deg",
            position="lower_left", font_size=10, color="white", name="hud",
        )
        pl.subplot(0, 1)
        viz.update_pov_camera(pl, state.turret)
        pl.render()

    if not off_screen:
        _add_widgets(pl, state.params)

    return pl, state, tick


def _add_widgets(pl, params: LoopParams) -> None:
    """Attach the live-tuning sliders + reference toggle to the world subplot."""
    pl.subplot(0, 0)

    def _slider(callback, rng, value, title, y):
        pl.add_slider_widget(
            callback, rng, value=value, title=title,
            pointa=(0.03, y), pointb=(0.32, y), style="modern",
            title_height=0.02, fmt="%.2f",
        )

    _slider(lambda v: setattr(params, "kp", v), [0.0, 20.0], params.kp, "Kp", 0.92)
    _slider(lambda v: setattr(params, "ki", v), [0.0, 20.0], params.ki, "Ki", 0.80)
    _slider(
        lambda v: setattr(params, "amplitude", np.radians(v)),
        [0.0, 45.0], np.degrees(params.amplitude), "Amplitude (deg)", 0.68,
    )
    _slider(
        lambda v: setattr(params, "frequency", v),
        [0.05, 2.0], params.frequency, "Frequency (Hz)", 0.56,
    )

    pl.add_checkbox_button_widget(
        lambda flag: setattr(params, "use_sine", flag),
        value=params.use_sine, position=(20, 20), size=28,
        color_on="#27ae60", color_off="#7f8c8d",
    )
    pl.add_text("Sine (else square)", position=(58, 22), font_size=9, color="white")


def main() -> None:
    pl, _state, tick = build_plotter()
    pl.add_callback(tick, interval=FRAME_INTERVAL_MS)
    pl.show(title="3D Turret Simulation")


if __name__ == "__main__":
    main()
