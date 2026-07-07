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

import numpy as np
import pyvista as pv

from control.control_system import ControlSystem, Mode
from simulation.turret_model import TurretModel
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
        """Write the buffer to ``path`` (CSV). ``channels`` selects column groups."""
        chans = [c for c in self.CHANNELS if c in channels]
        header = ["time_s", "mode"]
        if "error" in chans:
            header += ["error_unit", "az_error", "el_error"]
        if "control" in chans:
            header += ["az_control_deg_per_s", "el_control_deg_per_s"]
        if "disturbance" in chans:
            header += ["base_yaw_deg", "base_pitch_deg"]

        deg = np.degrees
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for (t, mode, is_rate, e_az, e_el, c_az, c_el, by, bp) in self.rows:
                out = [f"{t:.4f}", mode]
                if "error" in chans:
                    out += ["deg/s" if is_rate else "deg",
                            f"{deg(e_az):.5f}", f"{deg(e_el):.5f}"]
                if "control" in chans:
                    out += [f"{deg(c_az):.5f}", f"{deg(c_el):.5f}"]
                if "disturbance" in chans:
                    out += [f"{deg(by):.5f}", f"{deg(bp):.5f}"]
                w.writerow(out)
        return len(self.rows)


# --------------------------------------------------------------------------- #
# Simulation engine (GUI-independent, unit-tested).
# --------------------------------------------------------------------------- #
class SimEngine:
    """Owns the plant, disturbance and cascade controller, and steps the loop."""

    def __init__(self):
        self.turret = TurretModel(dt=DT, tau=0.2, rate_limit=RATE_LIMIT)
        self.board = TargetBoard(height_m=14.0, width=30.0, height_dim=24.0)
        self.stewart = StewartDisturbance()

        pivot = self.turret.base_position + np.array([0.0, 0.0, viz.PIVOT_HEIGHT])
        self.target_az, self.target_el = self.board.required_angles(pivot)
        self.control = ControlSystem(DT, RATE_LIMIT, el_hold=self.target_el)
        self.recorder = Recorder()

        self.t = 0.0
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.az_err = 0.0
        self.el_err = 0.0
        self.az_cmd = 0.0   # last commanded axis rate (control input, rad/s)
        self.el_cmd = 0.0
        # Rolling error history: (t, az_err, el_err) in SI (rad or rad/s by mode).
        maxlen = int(GRAPH_WINDOW / (DT * SUBSTEPS)) + 5
        self.history = collections.deque(maxlen=maxlen)

    def set_mode(self, mode: Mode) -> None:
        self.control.mode = mode
        self.control.reset()
        self.history.clear()

    def advance(self) -> None:
        """Step the closed loop by SUBSTEPS plant steps."""
        turret, cs = self.turret, self.control
        az_res = el_res = None
        for _ in range(SUBSTEPS):
            self.stewart.advance(DT)   # ramp the on/off envelope smoothly
            by, bp = self.stewart.angles(self.t)
            los_az, los_el = viz.los_angles(turret.azimuth, turret.elevation, by, bp)
            az_res, el_res = cs.step(
                self.t, los_az, turret.azimuth_rate, los_el, turret.elevation_rate,
                self.target_az, self.target_el,
            )
            turret.step(az_res.rate_cmd, el_res.rate_cmd)
            self.t += DT
            self.base_yaw, self.base_pitch = by, bp
        self.az_err, self.el_err = az_res.error, el_res.error
        self.az_cmd, self.el_cmd = az_res.rate_cmd, el_res.rate_cmd
        self.history.append((self.t, self.az_err, self.el_err))
        self.recorder.record((
            self.t, int(cs.mode), cs.error_is_rate, self.az_err, self.el_err,
            self.az_cmd, self.el_cmd, self.base_yaw, self.base_pitch,
        ))


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
    reticle = pl.add_text("+", position=(0.487, 0.472), viewport=True,
                          font_size=26, color="#efe36a")
    reticle.GetTextProperty().SetBold(True)
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

    w = QtWidgets.QWidget()
    w.setWindowTitle("Turret Control")
    w.resize(560, 760)
    root = QtWidgets.QVBoxLayout(w)

    # --- Mode + signal selectors ---
    form = QtWidgets.QFormLayout()
    mode_box = QtWidgets.QComboBox()
    mode_box.addItems(["1 - Speed loop", "2 - Position reference", "3 - Target tracking"])
    signal_box = QtWidgets.QComboBox()
    signal_box.addItems(["square", "sine", "constant"])
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
    cl.addWidget(FloatSlider("Speed Kp (inner)", 0.0, 20.0, cs.kp_speed, 0.1,
                             lambda v: setattr(cs, "kp_speed", v)))
    cl.addWidget(FloatSlider("Speed Ki (inner)", 0.0, 20.0, cs.ki_speed, 0.1,
                             lambda v: setattr(cs, "ki_speed", v)))
    cl.addWidget(amp)
    cl.addWidget(FloatSlider("Frequency", 0.05, 2.0, cs.frequency, 0.05,
                             lambda v: setattr(cs, "frequency", v), "Hz"))
    root.addWidget(ctl)

    # --- Disturbance group ---
    dis = QtWidgets.QGroupBox("Stewart platform disturbance")
    dl = QtWidgets.QVBoxLayout(dis)
    dist_btn = QtWidgets.QPushButton()
    dist_btn.setCheckable(True)
    dist_btn.setChecked(stew.enabled)
    dist_btn.setMaximumWidth(140)

    def _toggle_dist():
        stew.enabled = dist_btn.isChecked()
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
    status = QtWidgets.QLabel("idle")
    btn_row.addWidget(rec_btn)
    btn_row.addWidget(save_btn)
    btn_row.addWidget(status, 1)
    ll.addLayout(btn_row)
    root.addWidget(log)

    def _toggle_record():
        if rec_btn.isChecked():
            rec.start()
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

    def _refresh_status():
        if rec.recording:
            status.setText(f"recording… {len(rec.rows)} rows")
        elif rec.rows:
            status.setText(f"stopped ({len(rec.rows)} rows)")
        else:
            status.setText("idle")

    rec_btn.toggled.connect(lambda _c: _toggle_record())
    save_btn.clicked.connect(lambda: _save())

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
    root.addWidget(QtWidgets.QLabel("Error signal"))
    root.addWidget(canvas, 1)

    def _apply_mode():
        idx = mode_box.currentIndex()
        engine.set_mode(Mode(idx + 1))
        is_speed = engine.control.mode == Mode.SPEED
        is_target = engine.control.mode == Mode.TARGET
        amp.set_unit("deg/s" if is_speed else "deg")
        signal_box.setEnabled(not is_target)   # target derives its own reference
        amp.setEnabled(not is_target)

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
    _add_keyboard_controls(pl, engine, pov_fov)

    panel, update_graph = _make_control_panel(engine)
    panel.show()

    frame = [0]

    def tick(*_args):
        engine.advance()
        pl.subplot(0, 0)
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
