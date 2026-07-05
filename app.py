"""
Streamlit entry point for the 3D Turret Simulation.

This is the only module that wires ``control`` and ``simulation`` together and
renders the UI. It keeps the business logic thin: the PI controllers and the
turret plant do the work; here we build the reference, run the closed loop, and
draw the results.

Loop structure (see CLAUDE.md): the reference signal is a commanded pointing
angle. Per axis:  error = reference - angle  ->  PI  ->  commanded rate  ->
turret integrates the rate.  Azimuth follows the selected square/sine
reference; elevation holds the setpoint needed to point at the board.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from control.pi_controller import PIController
from control.reference_signals import REFERENCE_SIGNALS
from simulation.turret_model import TurretModel
from simulation.target_board import TargetBoard


def run_simulation(
    *,
    signal_name,
    amplitude_rad,
    frequency,
    offset_rad,
    duty,
    elevation_setpoint,
    kp,
    ki,
    dt,
    duration,
    tau,
    rate_limit,
):
    """Run the closed loop and return per-step arrays for plotting/rendering."""
    signal = REFERENCE_SIGNALS[signal_name]
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt

    if signal_name == "square":
        az_ref = signal(t, amplitude_rad, frequency, duty=duty, offset=offset_rad)
    else:
        az_ref = signal(t, amplitude_rad, frequency, offset=offset_rad)
    el_ref = np.full(n, elevation_setpoint)

    turret = TurretModel(dt=dt, tau=tau, rate_limit=rate_limit)
    az_ctrl = PIController(kp, ki, dt, output_limits=(-rate_limit, rate_limit))
    el_ctrl = PIController(kp, ki, dt, output_limits=(-rate_limit, rate_limit))

    az = np.zeros(n)
    el = np.zeros(n)
    for k in range(n):
        az[k] = turret.azimuth
        el[k] = turret.elevation
        az_cmd = az_ctrl.step(az_ref[k] - turret.azimuth)
        el_cmd = el_ctrl.step(el_ref[k] - turret.elevation)
        turret.step(az_cmd, el_cmd)

    return {"t": t, "az_ref": az_ref, "el_ref": el_ref, "az": az, "el": el}


def main() -> None:
    st.set_page_config(page_title="3D Turret Simulation", layout="wide")
    st.title("3D Turret Simulation")
    st.caption(
        "2-axis turret tracking a static target board. The reference drives "
        "azimuth; elevation holds on the board. Tune the loop live."
    )

    sb = st.sidebar
    sb.header("Reference")
    signal_name = sb.selectbox("Signal", list(REFERENCE_SIGNALS), index=0)
    amplitude_deg = sb.slider("Amplitude (deg)", 0.0, 45.0, 15.0, 0.5)
    frequency = sb.slider("Frequency (Hz)", 0.05, 2.0, 0.3, 0.05)
    offset_deg = sb.slider("Offset / nominal bearing (deg)", -30.0, 30.0, 0.0, 1.0)
    duty = sb.slider("Square duty", 0.1, 0.9, 0.5, 0.05) if signal_name == "square" else 0.5
    el_setpoint_deg = sb.slider("Elevation setpoint (deg)", -20.0, 20.0, 0.0, 0.5)

    sb.header("Controller (PI)")
    kp = sb.slider("Kp", 0.0, 20.0, 6.0, 0.1)
    ki = sb.slider("Ki", 0.0, 20.0, 2.0, 0.1)
    rate_limit_deg = sb.slider("Rate limit (deg/s)", 5.0, 180.0, 60.0, 5.0)

    sb.header("Plant & sim")
    tau = sb.slider("Rate time constant τ (s)", 0.01, 1.0, 0.2, 0.01)
    dt = sb.slider("Timestep dt (s)", 0.005, 0.05, 0.01, 0.005)
    duration = sb.slider("Duration (s)", 1.0, 20.0, 8.0, 0.5)

    board = TargetBoard()

    result = run_simulation(
        signal_name=signal_name,
        amplitude_rad=np.radians(amplitude_deg),
        frequency=frequency,
        offset_rad=np.radians(offset_deg),
        duty=duty,
        elevation_setpoint=np.radians(el_setpoint_deg),
        kp=kp,
        ki=ki,
        dt=dt,
        duration=duration,
        tau=tau,
        rate_limit=np.radians(rate_limit_deg),
    )

    t = result["t"]
    snapshot_t = st.slider("Snapshot time (s)", 0.0, float(t[-1]), float(t[-1]), float(dt))
    idx = int(round(snapshot_t / dt))
    idx = min(idx, len(t) - 1)

    # Rebuild a turret at the snapshot orientation for the 3D/POV renders.
    from simulation.visualization import draw_world_view, draw_pov_view

    turret = TurretModel(dt=dt, tau=tau)
    turret.azimuth = result["az"][idx]
    turret.elevation = result["el"][idx]

    col_world, col_pov = st.columns(2)
    with col_world:
        fig_w = plt.figure(figsize=(6, 5))
        ax_w = fig_w.add_subplot(111, projection="3d")
        draw_world_view(ax_w, turret, board)
        st.pyplot(fig_w)
        plt.close(fig_w)
    with col_pov:
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        draw_pov_view(ax_p, turret, board)
        st.pyplot(fig_p)
        plt.close(fig_p)

    st.subheader("Closed-loop response")
    fig_r, (ax_az, ax_el) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax_az.plot(t, np.degrees(result["az_ref"]), "--", color="gray", label="reference")
    ax_az.plot(t, np.degrees(result["az"]), color="tab:blue", label="azimuth")
    ax_az.axvline(snapshot_t, color="tab:orange", linewidth=1.0)
    ax_az.set_ylabel("Azimuth (deg)")
    ax_az.legend(loc="upper right", fontsize=8)
    ax_az.grid(True, alpha=0.3)

    ax_el.plot(t, np.degrees(result["el_ref"]), "--", color="gray", label="reference")
    ax_el.plot(t, np.degrees(result["el"]), color="tab:green", label="elevation")
    ax_el.axvline(snapshot_t, color="tab:orange", linewidth=1.0)
    ax_el.set_ylabel("Elevation (deg)")
    ax_el.set_xlabel("Time (s)")
    ax_el.legend(loc="upper right", fontsize=8)
    ax_el.grid(True, alpha=0.3)
    st.pyplot(fig_r)
    plt.close(fig_r)


if __name__ == "__main__":
    main()
