# 3D Turret Simulation

## Project overview

A simulation of a 2-axis (azimuth/elevation) turret tracking a static target board. This is an early stage of a larger project; the current scope is intentionally limited.

Current scope:
- Turret base carries a Stewart-platform disturbance: sinusoidal yaw/pitch base
  motion (magnitude 3-15 deg, frequency 0.1-0.4 Hz) the controller must reject.
- Cascade control: an inner speed loop (PI) with an outer position loop (P, Kp 1-20)
  wrapped around it. All loops regulate the line of sight (base + gimbal), gyro-style.
- Three selectable control modes: 1 SPEED (speed reference, deg/s), 2 POSITION
  (degree reference incl. a constant setpoint), 3 TARGET (auto-aim: barrel/target
  position error feeds the position loop).
- Reference signals are square / sine / constant, tunable live.
- Line of sight targets a static board 400 m downrange.
- Visualization is a real-time 3D-rendered (PyVista/VTK) world view plus a turret-mounted camera POV.
- Controls live in a secondary Qt window (mode switch + all sliders) with a live
  error-signal graph; the main 3D window is kept uncluttered.

Out of scope for now (future work): base translation, multi-target tracking, sensor
noise, feedforward / state-space / MPC controllers, realistic ballistics.

## Architecture

The project is split into two concerns: control and simulation, glued together by a PyVista app.

```
Python_3DTurret/
├── app.py                       # Entry point: 3D window (PyVista) + Qt control panel; SimEngine
├── control/
│   ├── pi_controller.py         # Discrete PI controller w/ anti-windup (per-axis)
│   ├── control_system.py        # Cascade controller: outer position P + inner speed PI; modes
│   └── reference_signals.py     # square_wave(), sine_wave(), constant_wave()
├── simulation/
│   ├── turret_model.py          # Turret plant model (2-axis gimbal, first-order rate)
│   ├── target_board.py          # Static target board geometry/position (400 m)
│   ├── stewart_platform.py      # Base yaw/pitch sine disturbance (StewartDisturbance)
│   ├── assets.py                # Downloads/caches scenery textures + mountain mesh
│   └── visualization.py         # PyVista 3D world view + turret POV camera + scenery
├── assets/                      # Downloaded scenery (gitignored; fetched on first run)
├── tests/test_system.py         # System tests (uv run python tests/test_system.py)
├── docs/control_theory.md       # Control-theory derivation + auditor checklist
└── pyproject.toml               # uv-managed project + dependencies
```

Design intent:
- `control/` has no knowledge of 3D geometry — it only deals with error signals, gains, and reference generation. It should be testable with plain floats/arrays.
- `simulation/` owns the plant (turret dynamics) and the world geometry (target board position, line-of-sight math). It consumes controller output, it does not compute it.
- `app.py` is the only place that wires control + simulation together and renders UI. Keep it thin — business logic belongs in the two packages above.

Run the app with:

```
uv run python app.py
```

Two windows open: the main 3D window (world view + turret POV, decluttered) and a
secondary Qt control panel with the mode switch, all control + disturbance sliders,
a live error-signal graph (position error in deg, or speed error in deg/s, adapting
to the mode), and a data-logging group (channel checkboxes + Record/Save-CSV) that
exports error, control input and disturbance to a CSV — all in one magnitude unit
family (deg / deg·s⁻¹). They are bridged without embedding VTK in Qt: the VTK
animation timer steps the sim, updates the graph, and pumps the Qt event loop via
`processEvents` (see `app.SimEngine` / `app.Recorder` / `_make_control_panel`).

Keyboard moves the cameras — arrows orbit the world view, `z`/`x` zoom it, `c`
resets it, and `[`/`]` zoom the turret POV. The world-view orbit is roll-free
(yaw/pitch only), for both keyboard and mouse (terrain style).

Units: control code is SI (rad, rad/s); the panel converts slider values in
deg / deg/s / Hz at the boundary. The outer position gain Kp is unit-agnostic
(rad/s per rad). See `control/control_system.py`.

Tests: `uv run python tests/test_system.py` (PI + anti-windup, reference signals,
disturbance, LOS composition, unit conversions, the three control modes,
disturbance rejection, tree placement, headless render).

Scenery (ground grass texture, tree bark, and a distant mountain elevation
mesh) is downloaded on first run by `simulation/assets.py` — grass/bark from
Poly Haven (CC0) and the Mount St. Helens DEM via `pyvista.examples` — cached
under `assets/`. If a download fails (offline), the scene falls back to flat
colours and still runs.

## Tech stack

Managed entirely through `uv` — all dependencies live in this project's `pyproject.toml` and `.venv`, not in the user's global Python environment.

- **numpy** — array math, angle/vector operations.
- **scipy** — signal processing helpers if needed (e.g. filtering).
- **pyvista** (VTK) — real-time 3D rendering of the world view and turret POV.
- **pyside6** (Qt) — the secondary control-panel window (mode switch + sliders).
- **matplotlib** — the live error-signal graph embedded in the control panel.

Environment setup:
```
uv sync          # install dependencies from pyproject.toml
uv add <pkg>      # add a new dependency
uv run <cmd>      # run any command inside the project's venv
```

Note: dependencies were declared in `pyproject.toml` during scaffolding, but `uv sync` may need to be run on the user's machine to actually download/install packages (the setup environment used to scaffold this project had restricted network access to PyPI).

## Conventions

- Units: SI throughout (meters, radians, seconds) unless a variable name says otherwise (e.g. `_deg`).
- Angles: azimuth measured from the turret's forward reference direction (0 rad = facing the target board's nominal bearing), elevation measured from horizontal.
- Coordinate frame: turret base at world origin `(0, 0, 0)`; target board centered at `(400, 0, h)` where `h` is the board's height offset.
- Keep control code physics-agnostic (operates on scalars/arrays of error, not on 3D scene objects).
- Prefer small, testable functions over large stateful classes where possible; use classes only where state genuinely needs to persist across simulation steps (e.g. `PIController`, `TurretModel`).

## Working with Claude on this project

- **When facing an ambiguous design or implementation choice, ask the user rather than guessing.** This includes: controller tuning defaults, slider ranges/step sizes, exact visualization style, coordinate/units conventions not already pinned down above, and whether a new dependency is worth adding.
- Prefer proposing 2-3 concrete options with trade-offs over open-ended questions.
- Since this project will grow (moving base, better controllers, more realistic dynamics are likely next), favor code that's easy to extend over premature optimization.
- If a task requires a new library, list the option(s) and get confirmation before running `uv add`.
