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
from control.reference_signals import square_wave, sine_wave  # noqa: E402
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


def test_tracking_no_disturbance():
    print("Tracking (no disturbance)")
    st = app._build_state()
    st.params.amplitude = 0.0           # aim straight at the board
    st.stewart.yaw_mag_deg = 0.0
    st.stewart.pitch_mag_deg = 0.0
    for _ in range(800):                # ~24 s
        app._advance(st)
    check("LOS settles on target", abs(np.degrees(st.az_err)) < 0.5,
          f"az err={np.degrees(st.az_err):.3f} deg")


def test_disturbance_rejection():
    print("Disturbance rejection")
    st = app._build_state()
    st.params.amplitude = 0.0
    st.params.kp, st.params.ki = 9.0, 3.0
    st.stewart.yaw_mag_deg, st.stewart.yaw_freq_hz = 10.0, 0.15
    st.stewart.pitch_mag_deg, st.stewart.pitch_freq_hz = 8.0, 0.2

    errs = []
    n = 900
    for k in range(n):
        app._advance(st)
        if k > n // 2:                  # measure after settling
            errs.append(st.az_err)
    closed_rms = float(np.sqrt(np.mean(np.square(errs))))
    # Open-loop (gimbal frozen) LOS-az error RMS equals the base-yaw RMS.
    open_rms = np.radians(st.stewart.yaw_mag_deg) / np.sqrt(2.0)
    atten = open_rms / max(closed_rms, 1e-9)
    check("controller attenuates the base disturbance", atten > 2.0,
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


def test_render_smoke():
    print("Headless render smoke")
    pl, st, tick = app.build_plotter(off_screen=True)
    st.stewart.yaw_mag_deg = 12.0
    for _ in range(30):
        tick()
    moved = abs(st.base_yaw) > 1e-6 or abs(st.base_pitch) > 1e-6
    check("simulation advanced under disturbance", st.t > 0 and moved,
          f"t={st.t:.2f}, base_yaw={np.degrees(st.base_yaw):.2f}")
    shot = os.path.join(tempfile.gettempdir(), "turret_smoke.png")
    pl.screenshot(shot)
    pl.close()
    check("screenshot rendered without error", os.path.exists(shot))


def main():
    for t in (test_pi_controller, test_reference_signals, test_disturbance,
              test_los_composition, test_tracking_no_disturbance,
              test_disturbance_rejection, test_tree_placement, test_render_smoke):
        t()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
