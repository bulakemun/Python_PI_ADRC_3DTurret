"""
System tests for the 3D turret simulation.

Runnable without any test framework:

    uv run python tests/test_system.py

Covers the control loop, the reference generators, the Stewart-platform
disturbance and its line-of-sight composition, closed-loop disturbance
rejection, tree placement (corridor sides / target cluster / turret clearing),
and a headless render smoke test. Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True

from control.pi_controller import PIController  # noqa: E402
from control.reference_signals import square_wave, sine_wave, constant_wave  # noqa: E402
from control.control_system import ControlSystem, Mode  # noqa: E402
from simulation.stewart_platform import StewartDisturbance  # noqa: E402
from simulation.target_board import TargetBoard  # noqa: E402
from simulation import visualization as viz  # noqa: E402
import app  # noqa: E402

_PASS, _FAIL = 0, 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# --------------------------------------------------------------------------- #
def test_pi_controller():
    print("PI controller")
    dt = 0.01
    pi = PIController(kp=2.0, ki=1.0, dt=dt, output_limits=(-1.0, 1.0))
    # Saturation is respected.
    check("output clamps to limits", pi.step(100.0) == 1.0)
    # Anti-windup pins the integral instead of letting it grow: while saturated
    # the integral term is back-calculated to (limit - kp*error) every step, so
    # it does not wind up (which would push ki*I toward +error*dt*N ~ +500 here).
    for _ in range(500):
        pi.step(100.0)
    expected = 1.0 - 2.0 * 100.0  # hi - kp*error
    check("anti-windup pins integral (no windup)",
          abs(pi.ki * pi.integral - expected) < 1.0,
          f"ki*I={pi.ki * pi.integral:.3f}, expected~{expected:.1f}")
    # First-order plant y' = u regulated to a setpoint converges.
    pi = PIController(kp=3.0, ki=1.5, dt=dt)
    y, sp = 0.0, 1.0
    for _ in range(4000):
        y += pi.step(sp - y) * dt
    check("regulates first-order plant to setpoint", abs(y - sp) < 1e-2,
          f"y={y:.4f}")


def test_reference_signals():
    print("Reference signals")
    t = np.linspace(0, 4, 4000)
    s = sine_wave(t, amplitude=2.0, frequency=0.5)
    check("sine amplitude", abs(s.max() - 2.0) < 1e-2 and abs(s.min() + 2.0) < 1e-2)
    sq = square_wave(t, amplitude=3.0, frequency=0.5)
    check("square is bipolar ±amplitude",
          set(np.round(np.unique(sq), 6)) == {-3.0, 3.0})
    c = constant_wave(t, amplitude=1.5, frequency=0.5)
    check("constant is a flat setpoint", np.allclose(c, 1.5))


def test_disturbance():
    print("Stewart disturbance")
    d = StewartDisturbance(yaw_mag_deg=10.0, yaw_freq_hz=0.2,
                           pitch_mag_deg=6.0, pitch_freq_hz=0.3)
    t = np.linspace(0, 20, 20000)
    yaw = np.array([d.yaw(tt) for tt in t])
    check("yaw magnitude matches slider", abs(np.degrees(yaw).max() - 10.0) < 0.1,
          f"max={np.degrees(yaw).max():.3f}")
    # Frequency via zero-crossings: 0.2 Hz over 20 s -> ~4 full cycles.
    zc = np.sum(np.diff(np.sign(yaw)) != 0)
    check("yaw frequency ~0.2 Hz", abs(zc / 2 / 20.0 - 0.2) < 0.02,
          f"freq={zc / 2 / 20.0:.3f}")
    # Analytic base rate (for the gyro feedback) matches a numeric derivative.
    numeric = (d.yaw(1.001) - d.yaw(0.999)) / 0.002
    check("yaw_rate matches numeric derivative",
          abs(d.yaw_rate(1.0) - numeric) < 1e-4, f"{d.yaw_rate(1.0):.5f} vs {numeric:.5f}")
    # Toggling off ramps a smooth envelope (no instantaneous jump), then stills.
    d.enabled = False
    d.advance(0.01)
    check("disable ramps smoothly (not an instant jump to 0)", abs(d.yaw(3.3)) > 0.0)
    for _ in range(60):                    # 0.6 s > 0.3 s ramp
        d.advance(0.01)
    check("disabled platform is still (magnitudes preserved)",
          abs(d.yaw(3.3)) < 1e-9 and abs(d.pitch(3.3)) < 1e-9 and d.yaw_mag_deg == 10.0)


def test_los_composition():
    print("Line-of-sight composition")
    # No base disturbance -> LOS equals the gimbal angles.
    az, el = viz.los_angles(0.3, 0.1, 0.0, 0.0)
    check("base=0 gives gimbal angles", abs(az - 0.3) < 1e-6 and abs(el - 0.1) < 1e-6)
    # Pure base yaw with zero gimbal -> LOS azimuth equals base yaw.
    az, el = viz.los_angles(0.0, 0.0, np.radians(8.0), 0.0)
    check("pure base yaw shows in LOS", abs(np.degrees(az) - 8.0) < 1e-6,
          f"az={np.degrees(az):.4f}")
    # base_tilt_matrix is a rigid transform (orthonormal rotation block).
    m = viz.base_tilt_matrix(np.radians(10.0), np.radians(5.0))
    r = m[:3, :3]
    check("base tilt is a rotation", np.allclose(r @ r.T, np.eye(3), atol=1e-9))


def _quiet_engine(mode):
    """A SimEngine with the base disturbance turned off, in the given mode."""
    e = app.SimEngine()
    e.stewart.yaw_mag_deg = 0.0
    e.stewart.pitch_mag_deg = 0.0
    e.set_mode(mode)
    return e


def test_unit_conversions():
    print("Unit conversions")
    cs = ControlSystem(dt=0.01, rate_limit=np.radians(180.0), el_hold=0.0)
    cs.amplitude_rad = np.radians(45.0)
    check("amplitude deg->rad", abs(cs.amplitude_rad - 0.7853981) < 1e-6)
    cs.mode = Mode.SPEED
    check("speed mode error is a rate", cs.error_is_rate is True)
    cs.mode = Mode.POSITION
    check("position mode error is a position", cs.error_is_rate is False)
    cs.signal = "constant"
    check("constant reference returns the amplitude",
          abs(cs.reference(3.7) - np.radians(45.0)) < 1e-9)


def test_mode_speed():
    print("Mode 1 - speed loop")
    e = _quiet_engine(Mode.SPEED)
    e.control.signal = "constant"
    e.control.amplitude_rad = np.radians(30.0)   # 30 deg/s speed reference
    for _ in range(600):                         # ~18 s
        e.advance()
    rate_deg = np.degrees(e.turret.azimuth_rate)
    check("axis speed tracks the speed reference", abs(rate_deg - 30.0) < 2.0,
          f"az rate={rate_deg:.2f} deg/s")


def test_mode_speed_stabilizes():
    print("Mode 1 - speed loop stabilises against the disturbance")
    e = app.SimEngine()
    e.set_mode(Mode.SPEED)
    e.control.signal = "constant"
    e.control.amplitude_rad = 0.0                # command LOS rate = 0 (hold)
    e.stewart.yaw_mag_deg, e.stewart.yaw_freq_hz = 10.0, 0.2
    e.stewart.pitch_mag_deg, e.stewart.pitch_freq_hz = 8.0, 0.25
    peak = 0.0
    for k in range(1500):                        # ~45 s
        e.advance()
        los_az, _ = viz.los_angles(e.turret.azimuth, e.turret.elevation,
                                   e.base_yaw, e.base_pitch)
        if k > 400:                              # after settling
            peak = max(peak, abs(np.degrees(los_az)))
    # Gyro rate feedback should reject the 10 deg base wobble to a few degrees;
    # without it the LOS would swing the full +/-10 deg (gimbal held, base free).
    check("speed loop holds the LOS against the base disturbance", peak < 3.0,
          f"LOS az peak={peak:.2f} deg (open-loop ~10 deg)")


def test_mode_position():
    print("Mode 2 - position reference")
    e = _quiet_engine(Mode.POSITION)
    e.control.signal = "constant"
    e.control.amplitude_rad = np.radians(12.0)   # 12 deg position reference
    for _ in range(800):
        e.advance()
    los_az, _ = viz.los_angles(e.turret.azimuth, e.turret.elevation, 0.0, 0.0)
    check("LOS settles at the position reference",
          abs(np.degrees(los_az) - 12.0) < 0.5, f"LOS az={np.degrees(los_az):.3f} deg")


def test_mode_target():
    print("Mode 3 - target tracking")
    e = _quiet_engine(Mode.TARGET)
    for _ in range(800):
        e.advance()
    los_az, los_el = viz.los_angles(e.turret.azimuth, e.turret.elevation, 0.0, 0.0)
    ok = (abs(los_az - e.target_az) < np.radians(0.5)
          and abs(los_el - e.target_el) < np.radians(0.5))
    check("LOS drives barrel/target error to zero", ok,
          f"az err={np.degrees(los_az - e.target_az):.3f}, "
          f"el err={np.degrees(los_el - e.target_el):.3f} deg")


def test_disturbance_rejection():
    print("Disturbance rejection (position mode)")
    e = app.SimEngine()
    e.set_mode(Mode.POSITION)
    e.control.signal = "constant"
    e.control.amplitude_rad = 0.0                # hold the board bearing
    e.control.kp_pos = 12.0
    e.stewart.yaw_mag_deg, e.stewart.yaw_freq_hz = 10.0, 0.15
    e.stewart.pitch_mag_deg, e.stewart.pitch_freq_hz = 8.0, 0.2

    errs = []
    n = 1200
    for k in range(n):
        e.advance()
        if k > n // 2:
            errs.append(e.az_err)
    closed_rms = float(np.sqrt(np.mean(np.square(errs))))
    open_rms = np.radians(e.stewart.yaw_mag_deg) / np.sqrt(2.0)  # gimbal frozen
    atten = open_rms / max(closed_rms, 1e-9)
    check("cascade attenuates the base disturbance", atten > 2.0,
          f"attenuation x{atten:.2f} (closed_rms={np.degrees(closed_rms):.2f} deg)")


def test_tree_placement():
    print("Tree placement")
    board = TargetBoard(height_m=14.0, width=30.0, height_dim=24.0)
    trunk, _foliage, _bark = viz._make_trees({}, board)
    p = trunk.points
    x, y = p[:, 0], p[:, 1]
    r = np.hypot(x, y)

    corridor = np.any((x > 45) & (x < 360) & (np.abs(y) >= 12) & (np.abs(y) <= 60))
    check("corridor-side trees exist between turret and target", corridor)

    around_target = np.any((x > 400) & (np.abs(y) < 90))
    check("trees exist around/behind the target", around_target)

    # Nothing directly in front of the board face (x just short of 400, near centre).
    in_front = np.any((x > 388) & (x < 399.5) & (np.abs(y) < 6))
    check("no trees in front of the target face", not in_front)

    # Turret clearing: no trunk points hugging the origin.
    near_origin = np.any(r < 20.0)
    check("turret clearing kept (no trees within 20 m)", not near_origin,
          f"min radius={r.min():.2f} m")


def test_recorder():
    print("Data recorder (CSV export)")
    e = app.SimEngine()
    e.set_mode(Mode.POSITION)
    e.recorder.start()
    for _ in range(20):
        e.advance()
    e.recorder.stop()
    check("recorder buffered rows", len(e.recorder.rows) == 20)

    path = os.path.join(tempfile.gettempdir(), "turret_log_test.csv")
    n = e.recorder.save(path, {"error", "control", "disturbance"})
    with open(path) as f:
        header = f.readline().strip().split(",")
        first = f.readline().strip().split(",")
    check("csv wrote all rows", n == 20)
    check("csv has error/control/disturbance columns in deg units",
          {"az_error", "az_control_deg_per_s", "base_yaw_deg"} <= set(header)
          and "deg" in first[2])

    # Channel selection ("menu"): disturbance-only export omits the others.
    e.recorder.save(path, {"disturbance"})
    with open(path) as f:
        hdr2 = f.readline().strip().split(",")
    check("channel selection limits columns",
          "base_yaw_deg" in hdr2 and "az_error" not in hdr2)


def test_camera_zoom_no_reset():
    print("Camera zoom does not reset the view")
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Sphere())
    r = pl.renderers[0]
    oc = app._OrbitController(r, (0.0, 0.0, 0.0))
    # Simulate a mouse-driven move to a known orientation (looking along -y).
    cam = r.GetActiveCamera()
    cam.SetFocalPoint(0.0, 0.0, 0.0)
    cam.SetPosition(0.0, -20.0, 0.0)
    r.ResetCameraClippingRange()
    oc.zoom(1.5)                            # keyboard zoom after the mouse move
    v = np.asarray(cam.GetPosition())
    az = np.degrees(np.arctan2(v[1], v[0]))
    check("zoom keeps the current orientation (no snap-back)",
          abs(az - (-90.0)) < 3.0 and np.linalg.norm(v) < 20.0,
          f"az={az:.1f} deg, dist={np.linalg.norm(v):.1f}")
    pl.close()


def test_camera_clamped_above_ground():
    print("Camera cannot look beneath the world plane")
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pv.Sphere())
    r = pl.renderers[0]
    focal = (0.0, 0.0, 1.35)
    oc = app._OrbitController(r, focal)
    cam = r.GetActiveCamera()
    cam.SetFocalPoint(*focal)
    cam.SetPosition(12.0, 0.0, -6.0)       # below the ground, looking up
    r.ResetCameraClippingRange()
    oc.clamp_view()
    p = np.asarray(cam.GetPosition())
    v = p - np.asarray(focal)
    el = np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))
    check("elevation clamped above the horizon and camera above ground",
          el >= 3.0 - 0.1 and p[2] > 0.0, f"el={el:.2f} deg, z={p[2]:.2f} m")
    pl.close()


def test_disturbance_disable_no_recenter():
    print("Disturbance disable preserves line of sight (speed mode)")
    e = app.SimEngine()
    e.set_mode(Mode.SPEED)
    e.control.signal = "constant"
    e.control.amplitude_rad = 0.0        # zero speed ref -> gimbal holds still
    e.turret.azimuth = 0.2                # nonzero so a "recenter" would be visible
    e.stewart.enabled = True
    e.stewart._env = 1.0                  # skip the ramp-in so LOS is settled
    for _ in range(50):
        e.advance()
    los_az_before, _ = viz.los_angles(e.turret.azimuth, e.turret.elevation,
                                      e.base_yaw, e.base_pitch)
    e.set_disturbance_enabled(False)
    for _ in range(10):
        e.advance()
    los_az_after, _ = viz.los_angles(e.turret.azimuth, e.turret.elevation,
                                     e.base_yaw, e.base_pitch)
    check("LOS az essentially unchanged after disabling disturbance",
          abs(np.degrees(los_az_after - los_az_before)) < 0.5,
          f"before={np.degrees(los_az_before):.3f}, after={np.degrees(los_az_after):.3f} deg")
    # Idempotent: disabling again (already off) must not fold a second time.
    az_before_second_disable = e.turret.azimuth
    e.set_disturbance_enabled(False)
    check("disabling twice is a no-op the second time",
          e.turret.azimuth == az_before_second_disable)


def test_recorder_stddev_footer():
    print("Recorder CSV stddev footer")
    e = app.SimEngine()
    e.set_mode(Mode.POSITION)
    e.stewart.yaw_mag_deg, e.stewart.pitch_mag_deg = 8.0, 5.0
    e.recorder.start()
    az_errs_deg = []
    for _ in range(30):
        e.advance()
        az_errs_deg.append(np.degrees(e.az_err))
    e.recorder.stop()

    path = os.path.join(tempfile.gettempdir(), "turret_log_stddev_test.csv")
    n = e.recorder.save(path, {"error", "control", "disturbance"})
    check("recorder saved 30 rows", n == 30)

    with open(path) as f:
        lines = [line.rstrip("\n") for line in f]
    header = lines[0].split(",")
    footer_row = None
    for line in lines:
        if line.startswith("stddev_1sigma"):
            footer_row = line.split(",")
            break
    check("csv has a stddev_1sigma footer row", footer_row is not None)
    if footer_row is not None:
        az_idx = header.index("az_error")
        expected = float(np.std(az_errs_deg))
        got = float(footer_row[az_idx])
        check("footer az_error stddev matches independently-computed std",
              abs(got - expected) < 1e-6,
              f"got={got:.8f}, expected={expected:.8f}")


def test_pov_reticle_centered():
    print("POV reticle centered on viewport")
    pl, scene, engine = app.build_scene(off_screen=True)
    hud = app._add_overlays(pl)
    engine.set_mode(Mode.TARGET)
    engine.stewart.enabled = False
    engine.stewart._env = 0.0
    for _ in range(200):
        engine.advance()
        scene.update(engine.turret, engine.base_yaw, engine.base_pitch)
        viz.update_pov_camera(pl, engine.turret, engine.base_yaw, engine.base_pitch)
    check("TARGET mode converges az error near zero",
          abs(np.degrees(engine.az_err)) < 1.0,
          f"az err={np.degrees(engine.az_err):.3f} deg")
    shot = os.path.join(tempfile.gettempdir(), "turret_pov_reticle.png")
    pl.screenshot(shot)
    check("screenshot with centered reticle rendered without error",
          os.path.exists(shot))

    reticle = [a for a in pl.renderers[1].actors.values()
              if hasattr(a, "GetTextProperty")]
    check("reticle actor present in POV subplot", len(reticle) >= 1)
    if reticle:
        import vtk
        tp = reticle[-1].GetTextProperty()
        check("reticle text is horizontally centered",
              tp.GetJustification() == vtk.VTK_TEXT_CENTERED)
        check("reticle text is vertically centered",
              tp.GetVerticalJustification() == vtk.VTK_TEXT_CENTERED)
    pl.close()


def test_speed_mode_target_error_toggle():
    print("Mode 1 - optional target-error readout")
    e = app.SimEngine()
    e.set_mode(Mode.SPEED)
    e.stewart.enabled = False
    e.stewart._env = 0.0
    e.control.speed_shows_target_error = True
    e.control.signal = "constant"
    e.control.amplitude_rad = 0.0        # zero speed ref -> turret stays put,
    for _ in range(20):                  # so LOS doesn't lag advance()'s
        e.advance()                      # internal sub-stepping
    check("error_is_rate is False when target-error readout is on",
          e.control.error_is_rate is False)
    los_az, _ = viz.los_angles(e.turret.azimuth, e.turret.elevation,
                               e.base_yaw, e.base_pitch)
    expected_err = e.target_az - los_az
    check("reported az_err is the target position error, not the speed error",
          abs(e.az_err - expected_err) < 1e-9,
          f"az_err={e.az_err:.6f}, expected={expected_err:.6f}")


def test_timed_recording_auto_save():
    print("Timed recording (Record 30 s -> CSV) auto-saves")
    e = app.SimEngine()
    e.set_mode(Mode.POSITION)
    path = os.path.join(tempfile.gettempdir(), "turret_log_timed_test.csv")
    if os.path.exists(path):
        os.remove(path)
    e.start_timed_recording(0.3, path, {"error", "control", "disturbance"})
    while e.t < 0.3:
        e.advance()
    check("auto-save wrote the CSV file", os.path.exists(path))
    check("engine.last_save_msg is set after auto-save", e.last_save_msg is not None,
          f"last_save_msg={e.last_save_msg!r}")


def test_render_smoke():
    print("Headless render smoke")
    pl, scene, engine = app.build_scene(off_screen=True)
    engine.set_mode(Mode.TARGET)
    engine.stewart.yaw_mag_deg = 12.0
    for _ in range(40):
        engine.advance()
        scene.update(engine.turret, engine.base_yaw, engine.base_pitch)
        viz.update_pov_camera(pl, engine.turret, engine.base_yaw, engine.base_pitch)
    moved = abs(engine.base_yaw) > 1e-6 or abs(engine.base_pitch) > 1e-6
    check("simulation advanced under disturbance", engine.t > 0 and moved,
          f"t={engine.t:.2f}, base_yaw={np.degrees(engine.base_yaw):.2f}")
    shot = os.path.join(tempfile.gettempdir(), "turret_smoke.png")
    pl.screenshot(shot)
    pl.close()
    check("screenshot rendered without error", os.path.exists(shot))


def main():
    for t in (test_pi_controller, test_reference_signals, test_disturbance,
              test_los_composition, test_unit_conversions, test_mode_speed,
              test_mode_speed_stabilizes, test_mode_position, test_mode_target,
              test_disturbance_rejection,
              test_recorder, test_camera_zoom_no_reset,
              test_camera_clamped_above_ground, test_tree_placement,
              test_disturbance_disable_no_recenter, test_recorder_stddev_footer,
              test_pov_reticle_centered, test_speed_mode_target_error_toggle,
              test_timed_recording_auto_save,
              test_render_smoke):
        t()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
