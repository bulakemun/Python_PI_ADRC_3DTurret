"""
Entry point for the 3D Turret Simulation (PyVista / VTK).

Opens an interactive window with two live 3D views -- a world view of the
turret model and a turret-POV camera -- with in-window sliders to tune the
control loop and keyboard controls to move the cameras. This is the only
module that wires ``control`` and ``simulation`` together; it keeps the logic
thin.

Per-axis loop (see CLAUDE.md):  error = reference - angle -> PI -> commanded
rate -> the turret integrates the rate. The reference signal drives azimuth;
elevation holds the angle that points at the board centre.

Keyboard (world view):  arrows = orbit, z / x = zoom, c = reset view.
Keyboard (turret POV):   [ / ] = zoom out / in.

Run with:
    uv run python app.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from control.pi_controller import PIController
from control.reference_signals import square_wave, sine_wave
from simulation.turret_model import TurretModel
from simulation.target_board import TargetBoard
from simulation.stewart_platform import StewartDisturbance
from simulation import visualization as viz
from simulation import assets as assets_mod

# Wall-clock pacing: advance this many plant steps per rendered frame so the
# animation runs at roughly real time (FRAME_INTERVAL_MS * SUBSTEPS ~= dt*1000).
DT = 0.01
SUBSTEPS = 3
FRAME_INTERVAL_MS = 30
RATE_LIMIT = np.radians(90.0)

# Keyboard camera-control increments.
ORBIT_DEG = 4.0
ZOOM_FACTOR = 1.1
POV_FOV_STEP = 4.0
POV_FOV_RANGE = (12.0, 70.0)

# Shared UI palette.
_TEXT_BG = (0.04, 0.07, 0.10)


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
    stewart: StewartDisturbance
    el_setpoint: float
    pov_fov: float = 40.0
    t: float = 0.0
    az_ref: float = 0.0
    az_err: float = 0.0
    base_yaw: float = 0.0
    base_pitch: float = 0.0


def _reference(params: LoopParams, t: float) -> float:
    """Current azimuth reference (rad) for the selected signal."""
    if params.use_sine:
        return float(sine_wave(t, params.amplitude, params.frequency))
    return float(square_wave(t, params.amplitude, params.frequency))


def _advance(state: SimState) -> None:
    """Step the closed loop by SUBSTEPS plant steps.

    The Stewart platform disturbs the base (yaw/pitch), so the controller
    regulates the *line of sight* (base + gimbal), stabilising the pointing
    against the disturbance rather than just the gimbal angle.
    """
    p = state.params
    state.az_ctrl.kp = state.el_ctrl.kp = p.kp
    state.az_ctrl.ki = state.el_ctrl.ki = p.ki
    az_err = state.az_err
    turret = state.turret
    for _ in range(SUBSTEPS):
        by, bp = state.stewart.angles(state.t)
        los_az, los_el = viz.los_angles(turret.azimuth, turret.elevation, by, bp)
        state.az_ref = _reference(p, state.t)
        az_err = state.az_ref - los_az
        el_err = state.el_setpoint - los_el
        turret.step(state.az_ctrl.step(az_err), state.el_ctrl.step(el_err))
        state.t += DT
        state.base_yaw, state.base_pitch = by, bp
    state.az_err = az_err


def _build_state() -> SimState:
    board = TargetBoard(height_m=14.0, width=30.0, height_dim=24.0)
    turret = TurretModel(dt=DT, tau=0.2, rate_limit=RATE_LIMIT)
    params = LoopParams()
    limits = (-RATE_LIMIT, RATE_LIMIT)
    # Elevation setpoint that points the line of sight at the board centre.
    pivot = turret.base_position + np.array([0.0, 0.0, viz.PIVOT_HEIGHT])
    _, el_setpoint = board.required_angles(pivot)
    return SimState(
        turret=turret,
        board=board,
        az_ctrl=PIController(params.kp, params.ki, DT, output_limits=limits),
        el_ctrl=PIController(params.kp, params.ki, DT, output_limits=limits),
        params=params,
        stewart=StewartDisturbance(),
        el_setpoint=el_setpoint,
    )


# --------------------------------------------------------------------------- #
# Text / widget styling helpers (legibility + opacity).
# --------------------------------------------------------------------------- #
def _style_text_prop(tp, bg_opacity: float = 0.55) -> None:
    """Make a VTK text property legible over the 3D scene: bold white with a
    shadow and a translucent dark background box."""
    tp.SetColor(1.0, 1.0, 1.0)
    tp.SetBold(True)
    tp.SetShadow(True)
    tp.SetFontFamilyToArial()
    tp.SetBackgroundColor(*_TEXT_BG)
    tp.SetBackgroundOpacity(bg_opacity)


def _style_slider(widget) -> None:
    """Enlarge and colour a slider so it reads clearly against the scene."""
    rep = widget.GetRepresentation()
    rep.SetTitleHeight(0.026)
    rep.SetLabelHeight(0.023)
    rep.SetTubeWidth(0.007)
    rep.SetSliderWidth(0.030)
    rep.SetSliderLength(0.022)
    rep.SetEndCapWidth(0.030)
    rep.SetEndCapLength(0.008)
    rep.GetTubeProperty().SetColor(0.55, 0.60, 0.66)
    rep.GetTubeProperty().SetOpacity(0.75)
    rep.GetSliderProperty().SetColor(0.16, 0.72, 0.92)   # handle
    rep.GetSelectedProperty().SetColor(0.22, 0.90, 0.55)
    rep.GetCapProperty().SetColor(0.80, 0.83, 0.88)
    rep.GetCapProperty().SetOpacity(0.85)
    _style_text_prop(rep.GetTitleProperty(), bg_opacity=0.6)
    _style_text_prop(rep.GetLabelProperty(), bg_opacity=0.6)


# --------------------------------------------------------------------------- #
# Roll-free orbit camera (pitch + yaw only).
# --------------------------------------------------------------------------- #
class _OrbitController:
    """Orbits a renderer's camera around a fixed focal point using spherical
    angles with the view-up pinned to world +z, so the camera never rolls -- it
    only yaws (azimuth) and pitches (elevation)."""

    def __init__(self, renderer, focal,
                 el_range=(np.radians(3.0), np.radians(85.0)),
                 dist_range=(4.0, 160.0)):
        self.r = renderer
        self.focal = np.asarray(focal, dtype=float)
        self.el_range = el_range
        self.dist_range = dist_range
        cam = renderer.GetActiveCamera()
        v = np.asarray(cam.GetPosition(), dtype=float) - self.focal
        self.dist = float(np.linalg.norm(v)) or 16.0
        self.az = float(np.arctan2(v[1], v[0]))
        self.el = float(np.arcsin(np.clip(v[2] / self.dist, -1.0, 1.0)))
        self._home = (self.az, self.el, self.dist)
        self.apply()

    def apply(self):
        el, az, d = self.el, self.az, self.dist
        offset = d * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        cam = self.r.GetActiveCamera()
        cam.SetPosition(*(self.focal + offset))
        cam.SetFocalPoint(*self.focal)
        cam.SetViewUp(0.0, 0.0, 1.0)  # world up -> no roll
        self.r.ResetCameraClippingRange()

    def orbit(self, d_az=0.0, d_el=0.0):
        self.az += d_az
        self.el = float(np.clip(self.el + d_el, *self.el_range))
        self.apply()

    def zoom(self, factor):
        self.dist = float(np.clip(self.dist / factor, *self.dist_range))
        self.apply()

    def reset(self):
        self.az, self.el, self.dist = self._home
        self.apply()


# --------------------------------------------------------------------------- #
# Widgets, overlays, keyboard.
# --------------------------------------------------------------------------- #
def _add_widgets(pl, state: SimState) -> None:
    """Attach the styled live-tuning sliders + reference toggle to the world view.

    Two columns: the control loop (Kp/Ki/reference) on the left, and the Stewart
    platform disturbance (yaw/pitch magnitude + frequency) on the right.
    """
    pl.subplot(0, 0)
    params, stewart = state.params, state.stewart

    def _slider(callback, rng, value, title, x0, x1, y):
        w = pl.add_slider_widget(
            callback, rng, value=value, title=title,
            pointa=(x0, y), pointb=(x1, y), style="modern",
            title_height=0.024, color="#e6edf3", fmt="%.2f",
        )
        _style_slider(w)

    # Left column -- control loop.
    lx0, lx1 = 0.035, 0.30
    _slider(lambda v: setattr(params, "kp", v), [0.0, 20.0], params.kp, "Kp",
            lx0, lx1, 0.90)
    _slider(lambda v: setattr(params, "ki", v), [0.0, 20.0], params.ki, "Ki",
            lx0, lx1, 0.78)
    _slider(lambda v: setattr(params, "amplitude", np.radians(v)),
            [0.0, 45.0], np.degrees(params.amplitude), "Amplitude (deg)",
            lx0, lx1, 0.66)
    _slider(lambda v: setattr(params, "frequency", v),
            [0.05, 2.0], params.frequency, "Frequency (Hz)", lx0, lx1, 0.54)

    # Right column -- Stewart platform disturbance (magnitude 3-15 deg, freq 0.1-0.4 Hz).
    rx0, rx1 = 0.37, 0.635
    _slider(lambda v: setattr(stewart, "yaw_mag_deg", v),
            [3.0, 15.0], stewart.yaw_mag_deg, "Yaw dist (deg)", rx0, rx1, 0.90)
    _slider(lambda v: setattr(stewart, "yaw_freq_hz", v),
            [0.1, 0.4], stewart.yaw_freq_hz, "Yaw freq (Hz)", rx0, rx1, 0.78)
    _slider(lambda v: setattr(stewart, "pitch_mag_deg", v),
            [3.0, 15.0], stewart.pitch_mag_deg, "Pitch dist (deg)", rx0, rx1, 0.66)
    _slider(lambda v: setattr(stewart, "pitch_freq_hz", v),
            [0.1, 0.4], stewart.pitch_freq_hz, "Pitch freq (Hz)", rx0, rx1, 0.54)

    # Placed under the sliders (well above the lower-left HUD) to avoid overlap.
    pl.add_checkbox_button_widget(
        lambda flag: setattr(params, "use_sine", flag),
        value=params.use_sine, position=(30, 345), size=30,
        color_on="#27ae60", color_off="#7f8c8d", border_size=2,
    )
    label = pl.add_text("Sine (else square)", position=(72, 349),
                        font_size=10, color="white")
    _style_text_prop(label.GetTextProperty(), bg_opacity=0.5)


def _add_keyboard_controls(pl, state: SimState) -> None:
    """Bind keys to move the world camera (roll-free yaw/pitch) and zoom the POV.
    Also switch the mouse interactor to terrain style, which keeps the view-up
    fixed so dragging the world view never rolls the camera."""
    world = pl.renderers[0]
    orbit_cam = _OrbitController(world, (0.0, 0.0, viz.PIVOT_HEIGHT))
    d = np.radians(ORBIT_DEG)

    def step(fn):
        fn()
        pl.render()

    def pov_zoom(delta):
        state.pov_fov = float(np.clip(state.pov_fov + delta, *POV_FOV_RANGE))
        pl.render()

    pl.add_key_event("Left", lambda: step(lambda: orbit_cam.orbit(d_az=+d)))
    pl.add_key_event("Right", lambda: step(lambda: orbit_cam.orbit(d_az=-d)))
    pl.add_key_event("Up", lambda: step(lambda: orbit_cam.orbit(d_el=+d)))
    pl.add_key_event("Down", lambda: step(lambda: orbit_cam.orbit(d_el=-d)))
    pl.add_key_event("z", lambda: step(lambda: orbit_cam.zoom(ZOOM_FACTOR)))
    pl.add_key_event("x", lambda: step(lambda: orbit_cam.zoom(1.0 / ZOOM_FACTOR)))
    pl.add_key_event("plus", lambda: step(lambda: orbit_cam.zoom(ZOOM_FACTOR)))
    pl.add_key_event("minus", lambda: step(lambda: orbit_cam.zoom(1.0 / ZOOM_FACTOR)))
    pl.add_key_event("c", lambda: step(orbit_cam.reset))
    pl.add_key_event("bracketright", lambda: pov_zoom(-POV_FOV_STEP))  # ] zoom in
    pl.add_key_event("bracketleft", lambda: pov_zoom(+POV_FOV_STEP))   # [ zoom out

    try:  # roll-free mouse navigation for the world view
        pl.enable_terrain_style(mouse_wheel_zooms=True)
    except Exception:  # pragma: no cover - backend dependent
        pass


def _add_overlays(pl, hud_text: str = "") -> object:
    """Add view titles, the POV reticle, and the controls legend. Returns the
    world-view HUD annotation so the animation loop can update it."""
    pl.subplot(0, 0)
    title_w = pl.add_text("World view", position="upper_edge", font_size=13,
                          color="white")
    _style_text_prop(title_w.GetTextProperty(), bg_opacity=0.5)
    help_txt = pl.add_text(
        "arrows: orbit   z / x: zoom   c: reset\n"
        "[ / ]: POV zoom   q: quit",
        position="lower_right", font_size=10, color="#dfe6ee",
    )
    _style_text_prop(help_txt.GetTextProperty(), bg_opacity=0.5)
    hud = pl.add_text(hud_text, position="lower_left", font_size=12,
                      color="white", name="hud")
    _style_text_prop(hud.GetTextProperty(), bg_opacity=0.6)

    pl.subplot(0, 1)
    title_p = pl.add_text("Turret POV", position="upper_edge", font_size=13,
                          color="white")
    _style_text_prop(title_p.GetTextProperty(), bg_opacity=0.5)
    reticle = pl.add_text("+", position=(0.487, 0.472), viewport=True,
                          font_size=26, color="#efe36a")
    reticle.GetTextProperty().SetBold(True)
    return hud


def build_plotter(off_screen: bool = False):
    """Assemble the two-view plotter, widgets, overlays, keyboard, and tick.

    Returns ``(plotter, state, tick)``. ``tick`` advances the simulation and
    refreshes both views; ``main`` drives it on a timer, and the headless test
    can call it directly.
    """
    state = _build_state()
    scenery = viz.build_environment(assets_mod.ensure_assets(), state.board)
    pl = pv.Plotter(shape=(1, 2), window_size=(1920, 1080), off_screen=off_screen)

    pl.subplot(0, 0)
    scene = viz.build_world_view(pl, state.turret, state.board, scenery)

    pl.subplot(0, 1)
    viz.build_pov_view(pl, state.turret, state.board, scenery)

    hud = _add_overlays(pl)

    try:
        pl.enable_anti_aliasing("fxaa")
    except Exception:  # pragma: no cover - backend dependent
        pass

    def tick(*_args) -> None:
        _advance(state)
        pl.subplot(0, 0)
        scene.update(state.turret, state.base_yaw, state.base_pitch)
        signal = "sine" if state.params.use_sine else "square"
        hud.SetText(
            0,
            f"Kp={state.params.kp:.1f}  Ki={state.params.ki:.1f}  ref={signal}\n"
            f"LOS az err={np.degrees(state.az_err):+5.1f} deg   "
            f"POV fov={state.pov_fov:.0f} deg\n"
            f"base disturbance: yaw={np.degrees(state.base_yaw):+5.1f}  "
            f"pitch={np.degrees(state.base_pitch):+5.1f} deg",
        )
        pl.subplot(0, 1)
        viz.update_pov_camera(pl, state.turret, state.base_yaw, state.base_pitch,
                              fov_deg=state.pov_fov)
        pl.render()

    _add_widgets(pl, state)
    _add_keyboard_controls(pl, state)

    return pl, state, tick


def main() -> None:
    pl, _state, tick = build_plotter()
    # Drive the animation on a repeating VTK timer (max_steps is effectively
    # "run for the whole session"; the timer fires every FRAME_INTERVAL_MS).
    pl.add_timer_event(max_steps=10_000_000, duration=FRAME_INTERVAL_MS, callback=tick)
    pl.show(title="3D Turret Simulation")


if __name__ == "__main__":
    main()
