"""
Entry point for the 3D Turret Simulation.

Two windows:
  * Main 3D window (PyVista/VTK): the decluttered world view + turret POV, with
    keyboard camera controls and a status HUD. No control sliders here.
  * Secondary control window (Qt/PySide6): the mode switch and every control /
    disturbance slider, plus a live error-signal graph.

The two are bridged without embedding VTK in Qt (which needs a real GL surface):
the VTK animation timer steps the simulation, refreshes the 3D scene, updates
the Qt graph, and pumps the Qt event loop via ``processEvents`` so the control
window stays responsive.

Control (see control/control_system.py): an outer position loop (P, Kp 1-20)
wraps the inner speed loop (PI). Modes: 1 SPEED (speed reference), 2 POSITION
(degree reference, incl. constant), 3 TARGET (auto-aim at the board). All loops
regulate the line of sight (Stewart-platform base disturbance + gimbal).

Run with:
    uv run python app.py
"""

from __future__ import annotations

import collections
import csv
import os
import time

import numpy as np
import pyvista as pv

from control.control_system import ControlSystem, Mode
from simulation.turret_model import TurretModel, TorqueTurretModel
from simulation.target_board import TargetBoard
from simulation.stewart_platform import StewartDisturbance
from simulation import visualization as viz
from simulation import assets as assets_mod

# Simulation pacing.
DT = 0.01
SUBSTEPS = 3
FRAME_INTERVAL_MS = 30
RATE_LIMIT = np.radians(180.0)          # plant rate command / speed-ref clamp

# Camera-control increments.
ORBIT_DEG = 4.0
ZOOM_FACTOR = 1.1
POV_FOV_STEP = 4.0
POV_FOV_RANGE = (12.0, 70.0)

# Error-graph rolling window (seconds).
GRAPH_WINDOW = 12.0
_TEXT_BG = (0.04, 0.07, 0.10)

# Torque plant parameters (SI). J_az > J_el: azimuth carries the whole gimbal.
TORQUE_PARAMS = dict(J_az=6.0, J_el=2.5, B=1.2, tau_c=0.8, K_t=0.8, i_max=50.0)

# Inner-loop gain defaults + slider ranges per plant. The inner PI output is a
# rate command (kinematic, [-]/[1/s] gains) or a current command (torque, gains
# in A per rad/s of rate error); retuned so the closed rate loop lands near the
# kinematic 0.2 s time constant. Values: (default, lo, hi, step).
PLANT_GAIN_UI = {
    "kinematic": {"kp": (6.0, 0.5, 20.0, 0.1), "ki": (2.0, 0.0, 20.0, 0.1)},
    "torque":    {"kp": (36.0, 0.0, 120.0, 1.0), "ki": (72.0, 0.0, 200.0, 1.0)},
}


# --------------------------------------------------------------------------- #
# Data recorder (error + control input + disturbance -> CSV).
# --------------------------------------------------------------------------- #
class Recorder:
    """Buffers per-step telemetry while recording and exports it as CSV/text.

    Everything is stored in SI and written in a single magnitude unit family:
    angles in degrees, rates in degrees/second -- so error, control input and
    disturbance are directly comparable. Which channels are written is chosen at
    export time.
    """

    #: Selectable channels -> the columns they contribute (deg / deg per s).
    CHANNELS = ("error", "control", "disturbance")

    def __init__(self):
        self.recording = False
        self.rows = []  # (t, mode, err_is_rate, err_az, err_el, cmd_az, cmd_el, by, bp) SI

    def start(self):
        self.rows = []
        self.recording = True

    def stop(self):
        self.recording = False

    def record(self, row):
        if self.recording:
            self.rows.append(row)

    def save(self, path, channels):
        """Write the buffer to ``path`` (CSV). ``channels`` selects column groups.

        After the data rows, appends a blank row and a ``stddev_1sigma`` footer
        row with the population standard deviation (ddof=0) of each numeric
        column, in the same deg / deg*s^-1 units as the data rows.
        """
        chans = [c for c in self.CHANNELS if c in channels]
        header = ["time_s", "mode"]
        if "error" in chans:
            header += ["error_unit", "az_error", "el_error"]
        if "control" in chans:
            header += ["az_control_deg_per_s", "el_control_deg_per_s"]
        if "disturbance" in chans:
            header += ["base_yaw_deg", "base_pitch_deg"]

        deg = np.degrees
        numeric_cols = collections.defaultdict(list)  # header name -> values (deg units)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for (t, mode, is_rate, e_az, e_el, c_az, c_el, by, bp) in self.rows:
                out = [f"{t:.4f}", mode]
                if "error" in chans:
                    err_unit = "deg/s" if is_rate else "deg"
                    az_deg, el_deg = deg(e_az), deg(e_el)
                    out += [err_unit, f"{az_deg:.5f}", f"{el_deg:.5f}"]
                    numeric_cols["az_error"].append(az_deg)
                    numeric_cols["el_error"].append(el_deg)
                if "control" in chans:
                    az_c, el_c = deg(c_az), deg(c_el)
                    out += [f"{az_c:.5f}", f"{el_c:.5f}"]
                    numeric_cols["az_control_deg_per_s"].append(az_c)
                    numeric_cols["el_control_deg_per_s"].append(el_c)
                if "disturbance" in chans:
                    by_deg, bp_deg = deg(by), deg(bp)
                    out += [f"{by_deg:.5f}", f"{bp_deg:.5f}"]
                    numeric_cols["base_yaw_deg"].append(by_deg)
                    numeric_cols["base_pitch_deg"].append(bp_deg)
                w.writerow(out)

            w.writerow([])
            footer = ["" for _ in header]
            footer[0] = "stddev_1sigma"
            for i, col in enumerate(header):
                if col in numeric_cols:
                    # More decimals than the per-row columns: this is a single
                    # summary value, so precision cost is negligible and it
                    # avoids compounding the per-row rounding.
                    footer[i] = f"{np.std(numeric_cols[col]):.8f}"
            w.writerow(footer)
        return len(self.rows)


# --------------------------------------------------------------------------- #
# Simulation engine (GUI-independent, unit-tested).
# --------------------------------------------------------------------------- #
class SimEngine:
    """Owns the plant, disturbance and cascade controller, and steps the loop."""

    def __init__(self, plant: str = "kinematic"):
        self.board = TargetBoard(height_m=14.0, width=30.0, height_dim=24.0)
        self.stewart = StewartDisturbance()

        base = np.array([0.0, 0.0, 0.0])
        pivot = base + np.array([0.0, 0.0, viz.PIVOT_HEIGHT])
        self.target_az, self.target_el = self.board.required_angles(pivot)
        self.control = ControlSystem(DT, RATE_LIMIT, el_hold=self.target_el)
        self.recorder = Recorder()

        # Reproducible measurement-noise generator (measurement noise only; the
        # true state used for rendering/recording stays clean). Defaults OFF.
        self.rng = np.random.default_rng(0)
        self.noise_enabled = False
        self.angle_noise_std = 0.0   # rad
        self.rate_noise_std = 0.0    # rad/s

        self.t = 0.0
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.az_err = 0.0
        self.el_err = 0.0
        self.az_cmd = 0.0   # last inner-loop command (rate rad/s, or current A)
        self.el_cmd = 0.0
        # Rolling error history: (t, az_err, el_err) in SI (rad or rad/s by mode).
        maxlen = int(GRAPH_WINDOW / (DT * SUBSTEPS)) + 5
        self.history = collections.deque(maxlen=maxlen)

        # Timed-recording ("Record 30 s -> CSV") state.
        self._auto_stop_t = None
        self._auto_save_path = None
        self._auto_channels = None
        self.last_save_msg = None

        self.plant_kind = None
        self.set_plant(plant)   # builds self.turret + inner-loop limits/gains

    def set_plant(self, kind: str) -> None:
        """Select the plant model. Resets plant state + controller integrators."""
        gains = PLANT_GAIN_UI[kind]
        if kind == "torque":
            self.turret = TorqueTurretModel(dt=DT, rate_limit=RATE_LIMIT,
                                            **TORQUE_PARAMS)
            self.control.set_inner_limits(self.turret.i_max_az, self.turret.i_max_el)
        else:
            self.turret = TurretModel(dt=DT, tau=0.2, rate_limit=RATE_LIMIT)
            self.control.set_inner_limits(RATE_LIMIT, RATE_LIMIT)
        self.control.kp_speed = gains["kp"][0]
        self.control.ki_speed = gains["ki"][0]
        self.plant_kind = kind
        self.control.reset()
        self.history.clear()
        self.base_yaw = self.base_pitch = 0.0
        self.az_err = self.el_err = self.az_cmd = self.el_cmd = 0.0

    def set_mode(self, mode: Mode) -> None:
        self.control.mode = mode
        self.control.reset()
        self.history.clear()

    def start_timed_recording(self, duration, path, channels) -> None:
        """Start recording now; auto-stop and auto-save after ``duration`` s."""
        self.recorder.start()
        self._auto_stop_t = self.t + duration
        self._auto_save_path = path
        self._auto_channels = channels
        self.last_save_msg = None

    def set_disturbance_enabled(self, on: bool) -> None:
        """Enable/disable the Stewart-platform base disturbance.

        Disabling folds the current base angle into the gimbal so the line of
        sight doesn't jump (this matters most in SPEED mode, where there is no
        position feedback to pull the barrel back: without the fold, the barrel
        would appear to "recenter" the instant the base disturbance vanishes).
        """
        if not on and self.stewart.enabled:
            by, bp = self.stewart.angles(self.t)
            self.turret.azimuth += by
            self.turret.elevation += bp
            self.stewart.snap(0.0)   # base angle -> 0 immediately (folded above)
            self.base_yaw = self.base_pitch = 0.0
        elif on and not self.stewart.enabled:
            # Start the sinusoids from a zero crossing so enabling doesn't inject
            # a base-angle step the SPEED-mode rate loop could never undo.
            self.stewart.restart(self.t)
        self.stewart.enabled = on

    def advance(self) -> None:
        """Step the closed loop by SUBSTEPS plant steps."""
        turret, cs = self.turret, self.control
        az_res = el_res = None
        for _ in range(SUBSTEPS):
            self.stewart.advance(DT)   # ramp the on/off envelope smoothly
            by, bp = self.stewart.angles(self.t)
            byr, bpr = self.stewart.rates(self.t)
            # True line of sight (angle + absolute gyro rate = gimbal + base).
            los_az, los_el = viz.los_angles(turret.azimuth, turret.elevation, by, bp)
            los_rate_az = turret.azimuth_rate + byr
            los_rate_el = turret.elevation_rate + bpr
            # Measurement noise feeds ONLY the controller; truth stays clean.
            if self.noise_enabled:
                m_az = los_az + self.rng.normal(0.0, self.angle_noise_std)
                m_el = los_el + self.rng.normal(0.0, self.angle_noise_std)
                m_raz = los_rate_az + self.rng.normal(0.0, self.rate_noise_std)
                m_rel = los_rate_el + self.rng.normal(0.0, self.rate_noise_std)
            else:
                m_az, m_el, m_raz, m_rel = los_az, los_el, los_rate_az, los_rate_el
            az_res, el_res = cs.step(
                self.t, m_az, m_raz, m_el, m_rel, self.target_az, self.target_el,
            )
            turret.step(az_res.rate_cmd, el_res.rate_cmd)
            self.t += DT
            self.base_yaw, self.base_pitch = by, bp
            # Record the TRUE error (reference vs clean state), not the noisy one.
            if cs.error_is_rate:
                self.az_err = az_res.reference - los_rate_az
                self.el_err = el_res.reference - los_rate_el
            else:
                self.az_err = az_res.reference - los_az
                self.el_err = el_res.reference - los_el
        self.az_cmd, self.el_cmd = az_res.rate_cmd, el_res.rate_cmd
        self.history.append((self.t, self.az_err, self.el_err))
        self.recorder.record((
            self.t, int(cs.mode), cs.error_is_rate, self.az_err, self.el_err,
            self.az_cmd, self.el_cmd, self.base_yaw, self.base_pitch,
        ))
        if self._auto_stop_t is not None and self.t >= self._auto_stop_t:
            self.recorder.stop()
            n = self.recorder.save(self._auto_save_path, self._auto_channels)
            self.last_save_msg = f"saved {n} rows -> {os.path.basename(self._auto_save_path)}"
            self._auto_stop_t = None


# --------------------------------------------------------------------------- #
# Text / overlay styling.
# --------------------------------------------------------------------------- #
def _style_text_prop(tp, bg_opacity: float = 0.55) -> None:
    tp.SetColor(1.0, 1.0, 1.0)
    tp.SetBold(True)
    tp.SetShadow(True)
    tp.SetFontFamilyToArial()
    tp.SetBackgroundColor(*_TEXT_BG)
    tp.SetBackgroundOpacity(bg_opacity)


# --------------------------------------------------------------------------- #
# Roll-free orbit camera (pitch + yaw only).
# --------------------------------------------------------------------------- #
class _OrbitController:
    """Orbits a renderer's camera with the view-up pinned to world +z (no roll)."""

    def __init__(self, renderer, focal,
                 el_range=(np.radians(3.0), np.radians(85.0)),
                 dist_range=(4.0, 160.0)):
        self.r = renderer
        self.focal = np.asarray(focal, dtype=float)
        self.el_range = el_range
        self.dist_range = dist_range
        self._read_camera()
        self._home = (self.az, self.el, self.dist)
        self.apply()

    def _read_camera(self):
        """Derive az/el/dist from the live camera (respects mouse-driven moves)."""
        cam = self.r.GetActiveCamera()
        v = np.asarray(cam.GetPosition(), dtype=float) - self.focal
        d = float(np.linalg.norm(v))
        if d > 1e-6:
            self.dist = d
            self.az = float(np.arctan2(v[1], v[0]))
            self.el = float(np.arcsin(np.clip(v[2] / d, -1.0, 1.0)))

    def apply(self):
        el, az, d = self.el, self.az, self.dist
        offset = d * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        cam = self.r.GetActiveCamera()
        cam.SetPosition(*(self.focal + offset))
        cam.SetFocalPoint(*self.focal)
        cam.SetViewUp(0.0, 0.0, 1.0)
        self.r.ResetCameraClippingRange()

    def orbit(self, d_az=0.0, d_el=0.0):
        self._read_camera()   # incorporate any mouse-driven change first
        self.az += d_az
        self.el = float(np.clip(self.el + d_el, *self.el_range))
        self.apply()

    def zoom(self, factor):
        self._read_camera()   # so zooming keeps the current view, not a stale one
        self.dist = float(np.clip(self.dist / factor, *self.dist_range))
        self.apply()

    def reset(self):
        self.az, self.el, self.dist = self._home
        self.apply()

    def clamp_view(self):
        """Re-read the live camera and pin it above the ground-plane horizon.

        Called every frame so mouse (terrain-style) navigation can't swing the
        camera below the world plane to look up from underneath. It is a no-op
        while the view is already within the elevation/distance limits.
        """
        self._read_camera()
        el = float(np.clip(self.el, *self.el_range))
        dist = float(np.clip(self.dist, *self.dist_range))
        if el != self.el or dist != self.dist:
            self.el, self.dist = el, dist
            self.apply()


def _add_overlays(pl):
    """View titles, POV reticle and controls legend. Returns the HUD annotation."""
    pl.subplot(0, 0)
    title_w = pl.add_text("World view", position="upper_edge", font_size=13,
                          color="white")
    _style_text_prop(title_w.GetTextProperty(), bg_opacity=0.5)
    help_txt = pl.add_text(
        "arrows: orbit   z / x: zoom   c: reset\n[ / ]: POV zoom   q: quit",
        position="lower_right", font_size=10, color="#dfe6ee",
    )
    _style_text_prop(help_txt.GetTextProperty(), bg_opacity=0.5)
    hud = pl.add_text("", position="lower_left", font_size=12, color="white",
                      name="hud")
    _style_text_prop(hud.GetTextProperty(), bg_opacity=0.6)

    pl.subplot(0, 1)
    title_p = pl.add_text("Turret POV", position="upper_edge", font_size=13,
                          color="white")
    _style_text_prop(title_p.GetTextProperty(), bg_opacity=0.5)
    reticle = pl.add_text("+", position=(0.5, 0.5), viewport=True,
                          font_size=26, color="#efe36a")
    rtp = reticle.GetTextProperty()
    rtp.SetBold(True)
    rtp.SetJustificationToCentered()
    rtp.SetVerticalJustificationToCentered()
    return hud


def _add_keyboard_controls(pl, engine: SimEngine, pov):
    """Roll-free world-camera keys + POV zoom; terrain-style mouse."""
    world = pl.renderers[0]
    orbit_cam = _OrbitController(world, (0.0, 0.0, viz.PIVOT_HEIGHT))
    d = np.radians(ORBIT_DEG)

    def step(fn):
        fn()
        pl.render()

    def pov_zoom(delta):
        pov[0] = float(np.clip(pov[0] + delta, *POV_FOV_RANGE))
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
    pl.add_key_event("bracketright", lambda: pov_zoom(-POV_FOV_STEP))
    pl.add_key_event("bracketleft", lambda: pov_zoom(+POV_FOV_STEP))

    try:
        pl.enable_terrain_style(mouse_wheel_zooms=True)
    except Exception:  # pragma: no cover - backend dependent
        pass

    return orbit_cam


def build_scene(off_screen: bool = False):
    """Build the two-view 3D plotter and scene. Returns (plotter, scene, engine)."""
    engine = SimEngine()
    scenery = viz.build_environment(assets_mod.ensure_assets(), engine.board)
    pl = pv.Plotter(shape=(1, 2), window_size=(1920, 1080), off_screen=off_screen)
    pl.subplot(0, 0)
    scene = viz.build_world_view(pl, engine.turret, engine.board, scenery)
    pl.subplot(0, 1)
    viz.build_pov_view(pl, engine.turret, engine.board, scenery)
    try:
        pl.enable_anti_aliasing("fxaa")
    except Exception:  # pragma: no cover
        pass
    return pl, scene, engine


# --------------------------------------------------------------------------- #
# Qt control panel (secondary window) with the live error graph.
# --------------------------------------------------------------------------- #
def _make_control_panel(engine: SimEngine):
    """Create the Qt control-panel window. Returns (widget, graph_updater)."""
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6 import QtWidgets, QtCore
    import matplotlib
    matplotlib.use("qtagg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvas

    cs, stew = engine.control, engine.stewart

    class FloatSlider(QtWidgets.QWidget):
        def __init__(self, name, lo, hi, val, step, cb, unit=""):
            super().__init__()
            self._lo, self._step, self._cb = lo, step, cb
            self._name, self._unit = name, unit
            lay = QtWidgets.QHBoxLayout(self)
            lay.setContentsMargins(2, 1, 2, 1)
            self._lbl = QtWidgets.QLabel()
            self._lbl.setMinimumWidth(150)
            self._s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self._s.setMinimum(0)
            self._s.setMaximum(int(round((hi - lo) / step)))
            self._s.setValue(int(round((val - lo) / step)))
            self._s.valueChanged.connect(self._on)
            lay.addWidget(self._lbl)
            lay.addWidget(self._s, 1)
            self._refresh()

        def _val(self):
            return self._lo + self._s.value() * self._step

        def _on(self, _):
            self._refresh()
            self._cb(self._val())

        def _refresh(self):
            self._lbl.setText(f"{self._name}: {self._val():.2f} {self._unit}")

        def set_unit(self, u):
            self._unit = u
            self._refresh()

        def reconfigure(self, lo, hi, val, step):
            """Retarget the slider's range/value (e.g. on a plant switch)."""
            self._lo, self._step = lo, step
            self._s.blockSignals(True)
            self._s.setMaximum(int(round((hi - lo) / step)))
            self._s.setValue(int(round((val - lo) / step)))
            self._s.blockSignals(False)
            self._refresh()

    w = QtWidgets.QWidget()
    w.setWindowTitle("Turret Control")
    w.resize(600, 940)
    outer = QtWidgets.QVBoxLayout(w)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    outer.addWidget(scroll)
    content = QtWidgets.QWidget()
    scroll.setWidget(content)
    root = QtWidgets.QVBoxLayout(content)

    # --- Plant + mode + signal selectors ---
    form = QtWidgets.QFormLayout()
    plant_box = QtWidgets.QComboBox()
    plant_box.addItems(["Kinematic (rate servo)", "Torque (J, B, K, friction)"])
    mode_box = QtWidgets.QComboBox()
    mode_box.addItems(["1 - Speed loop", "2 - Position reference", "3 - Target tracking"])
    signal_box = QtWidgets.QComboBox()
    signal_box.addItems(["square", "sine", "constant"])
    form.addRow("Plant", plant_box)
    form.addRow("Mode", mode_box)
    form.addRow("Reference signal", signal_box)
    root.addLayout(form)

    # --- Control-loop group ---
    ctl = QtWidgets.QGroupBox("Control loop")
    cl = QtWidgets.QVBoxLayout(ctl)
    amp = FloatSlider("Amplitude", 0.0, 90.0, np.degrees(cs.amplitude_rad), 1.0,
                      lambda v: setattr(cs, "amplitude_rad", np.radians(v)), "deg/s")
    cl.addWidget(FloatSlider("Kp (position, outer)", 1.0, 20.0, cs.kp_pos, 0.5,
                             lambda v: setattr(cs, "kp_pos", v)))
    # Inner-gain ranges/units depend on the plant (rate command vs current
    # command); reconfigured on a plant switch. Floored so the inner loop is
    # never fully gutted (Kp_s = 0 = dead/near-unstable).
    _kp_ui = PLANT_GAIN_UI[engine.plant_kind]["kp"]
    _ki_ui = PLANT_GAIN_UI[engine.plant_kind]["ki"]
    kp_sl = FloatSlider("Speed Kp (inner)", _kp_ui[1], _kp_ui[2], cs.kp_speed,
                        _kp_ui[3], lambda v: setattr(cs, "kp_speed", v))
    ki_sl = FloatSlider("Speed Ki (inner)", _ki_ui[1], _ki_ui[2], cs.ki_speed,
                        _ki_ui[3], lambda v: setattr(cs, "ki_speed", v))
    cl.addWidget(kp_sl)
    cl.addWidget(ki_sl)
    cl.addWidget(amp)
    cl.addWidget(FloatSlider("Frequency", 0.05, 2.0, cs.frequency, 0.05,
                             lambda v: setattr(cs, "frequency", v), "Hz"))
    show_target_err_chk = QtWidgets.QCheckBox("Show target error (speed mode)")
    show_target_err_chk.setChecked(cs.speed_shows_target_error)
    show_target_err_chk.toggled.connect(
        lambda c: setattr(cs, "speed_shows_target_error", c))
    cl.addWidget(show_target_err_chk)
    root.addWidget(ctl)

    # --- Disturbance group ---
    dis = QtWidgets.QGroupBox("Stewart platform disturbance")
    dl = QtWidgets.QVBoxLayout(dis)
    dist_btn = QtWidgets.QPushButton()
    dist_btn.setCheckable(True)
    dist_btn.setChecked(stew.enabled)
    dist_btn.setMaximumWidth(140)

    def _toggle_dist():
        engine.set_disturbance_enabled(dist_btn.isChecked())
        dist_btn.setText("Disturbance: ON" if stew.enabled else "Disturbance: OFF")

    dist_btn.toggled.connect(lambda _c: _toggle_dist())
    _toggle_dist()
    dl.addWidget(dist_btn)
    dl.addWidget(FloatSlider("Yaw magnitude", 3.0, 15.0, stew.yaw_mag_deg, 0.5,
                             lambda v: setattr(stew, "yaw_mag_deg", v), "deg"))
    dl.addWidget(FloatSlider("Yaw frequency", 0.1, 0.4, stew.yaw_freq_hz, 0.01,
                             lambda v: setattr(stew, "yaw_freq_hz", v), "Hz"))
    dl.addWidget(FloatSlider("Pitch magnitude", 3.0, 15.0, stew.pitch_mag_deg, 0.5,
                             lambda v: setattr(stew, "pitch_mag_deg", v), "deg"))
    dl.addWidget(FloatSlider("Pitch frequency", 0.1, 0.4, stew.pitch_freq_hz, 0.01,
                             lambda v: setattr(stew, "pitch_freq_hz", v), "Hz"))
    root.addWidget(dis)

    # --- Sensor noise group (measurement noise only; truth stays clean) ---
    noise = QtWidgets.QGroupBox("Sensor noise (on the controller's measurements)")
    nl = QtWidgets.QVBoxLayout(noise)
    noise_chk = QtWidgets.QCheckBox("Enable sensor noise")
    noise_chk.setChecked(engine.noise_enabled)
    noise_chk.toggled.connect(lambda c: setattr(engine, "noise_enabled", c))
    nl.addWidget(noise_chk)
    nl.addWidget(FloatSlider("Angle noise σ", 0.0, 0.5, np.degrees(engine.angle_noise_std),
                             0.01, lambda v: setattr(engine, "angle_noise_std",
                                                     np.radians(v)), "deg"))
    nl.addWidget(FloatSlider("Rate noise σ", 0.0, 2.0, np.degrees(engine.rate_noise_std),
                             0.05, lambda v: setattr(engine, "rate_noise_std",
                                                     np.radians(v)), "deg/s"))
    root.addWidget(noise)

    # --- Data logging group ---
    rec = engine.recorder
    log = QtWidgets.QGroupBox("Data logging (deg / deg·s⁻¹)")
    ll = QtWidgets.QVBoxLayout(log)
    # The "menu" of channels to export.
    chan_row = QtWidgets.QHBoxLayout()
    chk_err = QtWidgets.QCheckBox("Error")
    chk_ctl = QtWidgets.QCheckBox("Control input")
    chk_dis = QtWidgets.QCheckBox("Disturbance")
    for c in (chk_err, chk_ctl, chk_dis):
        c.setChecked(True)
        chan_row.addWidget(c)
    ll.addLayout(chan_row)

    btn_row = QtWidgets.QHBoxLayout()
    rec_btn = QtWidgets.QPushButton("Record")
    rec_btn.setCheckable(True)
    save_btn = QtWidgets.QPushButton("Save CSV…")
    quick_rec_btn = QtWidgets.QPushButton("Record 30 s -> CSV")
    status = QtWidgets.QLabel("idle")
    btn_row.addWidget(rec_btn)
    btn_row.addWidget(save_btn)
    btn_row.addWidget(quick_rec_btn)
    ll.addLayout(btn_row)
    ll.addWidget(status)
    root.addWidget(log)

    def _toggle_record():
        if rec_btn.isChecked():
            rec.start()
            engine.last_save_msg = None
            rec_btn.setText("Recording…")
        else:
            rec.stop()
            rec_btn.setText("Record")
        _refresh_status()

    def _selected_channels():
        sel = set()
        if chk_err.isChecked():
            sel.add("error")
        if chk_ctl.isChecked():
            sel.add("control")
        if chk_dis.isChecked():
            sel.add("disturbance")
        return sel

    def _save():
        if not rec.rows:
            status.setText("nothing recorded yet")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            w, "Save telemetry", "turret_log.csv", "CSV / text (*.csv *.txt)")
        if not path:
            return
        n = rec.save(path, _selected_channels())
        status.setText(f"saved {n} rows → {os.path.basename(path)}")

    def _record_30s():
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"turret_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        )
        engine.start_timed_recording(30.0, path, {"error", "control", "disturbance"})
        status.setText("recording 30 s…")

    def _refresh_status():
        if engine.last_save_msg:
            status.setText(engine.last_save_msg)
        elif rec.recording:
            status.setText(f"recording… {len(rec.rows)} rows")
        elif rec.rows:
            status.setText(f"stopped ({len(rec.rows)} rows)")
        else:
            status.setText("idle")

    rec_btn.toggled.connect(lambda _c: _toggle_record())
    save_btn.clicked.connect(lambda: _save())
    quick_rec_btn.clicked.connect(lambda: _record_30s())

    # --- Error graph ---
    fig = Figure(figsize=(5, 2.4), tight_layout=True)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    (line_az,) = ax.plot([], [], color="tab:blue", label="azimuth")
    (line_el,) = ax.plot([], [], color="tab:green", label="elevation")
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("time (s)")
    canvas.setMinimumHeight(230)
    root.addWidget(QtWidgets.QLabel("Error signal"))
    root.addWidget(canvas, 1)

    def _apply_plant():
        kind = "torque" if plant_box.currentIndex() == 1 else "kinematic"
        engine.set_plant(kind)   # sets default gains + resets state
        kp = PLANT_GAIN_UI[kind]["kp"]
        ki = PLANT_GAIN_UI[kind]["ki"]
        kp_sl.reconfigure(kp[1], kp[2], engine.control.kp_speed, kp[3])
        ki_sl.reconfigure(ki[1], ki[2], engine.control.ki_speed, ki[3])

    plant_box.currentIndexChanged.connect(lambda _i: _apply_plant())

    def _apply_mode():
        idx = mode_box.currentIndex()
        engine.set_mode(Mode(idx + 1))
        is_speed = engine.control.mode == Mode.SPEED
        is_target = engine.control.mode == Mode.TARGET
        amp.set_unit("deg/s" if is_speed else "deg")
        signal_box.setEnabled(not is_target)   # target derives its own reference
        amp.setEnabled(not is_target)
        show_target_err_chk.setEnabled(is_speed)

    def _apply_signal():
        cs.signal = signal_box.currentText()

    mode_box.currentIndexChanged.connect(lambda _i: _apply_mode())
    signal_box.currentIndexChanged.connect(lambda _i: _apply_signal())
    _apply_mode()

    def update_graph():
        _refresh_status()
        if not engine.history:
            return
        data = np.array(engine.history)
        t = data[:, 0]
        rate = engine.control.error_is_rate
        az = np.degrees(data[:, 1])
        el = np.degrees(data[:, 2])
        line_az.set_data(t, az)
        line_el.set_data(t, el)
        ax.set_xlim(max(0.0, t[-1] - GRAPH_WINDOW), max(GRAPH_WINDOW, t[-1]))
        lo = min(az.min(), el.min(), -1.0)
        hi = max(az.max(), el.max(), 1.0)
        pad = 0.1 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel("error (deg/s)" if rate else "error (deg)")
        ax.set_title(f"{'Speed' if rate else 'Position'} error")
        canvas.draw_idle()

    return w, update_graph


# --------------------------------------------------------------------------- #
def main() -> None:
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6 import QtWidgets

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    pl, scene, engine = build_scene(off_screen=False)
    hud = _add_overlays(pl)
    pov_fov = [40.0]
    orbit_cam = _add_keyboard_controls(pl, engine, pov_fov)

    panel, update_graph = _make_control_panel(engine)
    panel.show()

    frame = [0]

    def tick(*_args):
        engine.advance()
        pl.subplot(0, 0)
        orbit_cam.clamp_view()   # keep the world camera above the ground plane
        scene.update(engine.turret, engine.base_yaw, engine.base_pitch)
        cs = engine.control
        unit = "deg/s" if cs.error_is_rate else "deg"
        hud.SetText(
            0,
            f"mode {int(cs.mode)}: {cs.mode.name}   Kp={cs.kp_pos:.1f}  "
            f"sKp={cs.kp_speed:.1f}  sKi={cs.ki_speed:.1f}\n"
            f"az err={np.degrees(engine.az_err):+6.2f} {unit}   POV fov={pov_fov[0]:.0f} deg\n"
            f"base disturbance: yaw={np.degrees(engine.base_yaw):+5.1f}  "
            f"pitch={np.degrees(engine.base_pitch):+5.1f} deg",
        )
        pl.subplot(0, 1)
        viz.update_pov_camera(pl, engine.turret, engine.base_yaw, engine.base_pitch,
                              fov_deg=pov_fov[0])
        pl.render()
        frame[0] += 1
        if frame[0] % 3 == 0:          # ~10 fps graph refresh
            update_graph()
        qt_app.processEvents()         # keep the control window responsive

    pl.add_timer_event(max_steps=10_000_000, duration=FRAME_INTERVAL_MS, callback=tick)
    pl.show(title="3D Turret Simulation")


if __name__ == "__main__":
    main()
