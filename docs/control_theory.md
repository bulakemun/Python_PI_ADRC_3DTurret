# Control theory of the 3D turret simulation — reasoning for review

This document lays out the control-theoretic reasoning behind the simulation so
another model (or person) can audit it for correctness. It states the plant,
the disturbance, the line-of-sight kinematics, the controller structure, the
three modes, the unit bookkeeping, and — importantly — the **approximations and
claims that should be checked**. Where a symbol maps to code, the file is cited.

Everything internal is SI: angles in radians, rates in rad/s, time in seconds.
The GUI converts deg / deg·s⁻¹ / Hz at the boundary only.

---

## 1. What is being controlled

A 2-axis (azimuth `az`, elevation `el`) gimbal sits on a **Stewart platform**
whose base yaws and pitches sinusoidally (a disturbance). The controlled
quantity is the **line of sight (LOS)** — the barrel's absolute pointing in the
world, i.e. the composition of base motion and gimbal motion. The intent is
*gyro-style stabilization*: regulate the absolute LOS, so the barrel holds/tracks
its commanded pointing despite the moving base.

Per axis there is an **outer position loop** (P) wrapped around an **inner speed
loop** (PI), a classic cascaded servo. The inner loop closes on the **absolute
LOS rate** (what a rate gyro senses); the outer loop closes on the **absolute
LOS angle**.

---

## 2. Frames and sign conventions

- Turret base at world origin. Target board centred at `(400, 0, h)`, `h≈14 m`.
- Azimuth measured about world **+z** from the **+x** forward axis (0 = board bearing).
- Elevation measured from horizontal, **positive up**.
- Barrel unit vector from gimbal angles `(az, el)`:
  `d = (cos el cos az, cos el sin az, sin el)` — see `turret_model.py:barrel_direction`.
  Check: `el = +90°` ⇒ `d = (0,0,1)` (straight up). ✔ positive-up convention.

Rotation matrices (`visualization.py`):
- `Rz(a)` = yaw about z, `Ry(a)` = pitch about y.
- Base disturbance rotation `R_base = Rz(β_y)·Ry(−β_p)` (`_base_rotation`).
- Gimbal rotation `R_gim = Rz(g_az)·Ry(−g_el)`.
- The `Ry(−·)` sign makes positive pitch raise the nose (+z), consistent with §2.

---

## 3. Plant model (per axis)  — `simulation/turret_model.py`

The gimbal is a **first-order velocity servo**. Given a commanded rate `u`
(rad/s), the discrete update (semi-implicit Euler, `dt = 0.01 s`, `τ = 0.2 s`) is:

```
u        := clip(u, ±rate_limit)          # rate_limit = 180°/s
ω[k+1]   := ω[k] + (u[k] − ω[k])·dt/τ      # actual rate lags command
φ[k+1]   := φ[k] + ω[k+1]·dt               # angle integrates the rate
```

Continuous-time equivalent:

```
τ·ω̇ = u − ω        ⇒   ω(s)/u(s) = 1/(τs + 1)      (velocity servo, τ=0.2s ≈ 0.8 Hz)
φ̇ = ω              ⇒   φ(s)/u(s) = 1/[s(τs + 1)]
```

So the **plant the controllers see** is: *command a rate, get a lagged actual
rate, whose integral is the angle.* The gimbal angle/rate are `φ = g_az/g_el`,
`ω = ġ`. Elevation is hard-limited to `(−90°, +90°)` (rate zeroed at the stop).

> Note: this "plant" already includes an internal rate servo (the `τ` lag). The
> inner PI speed loop below therefore controls a first-order rate plant, and the
> outer P loop controls the resulting rate→angle integrator. **Check:** the
> inner loop is a rate loop around a rate servo (valid, just means the effective
> rate bandwidth is set by both `τ` and `Kp_s/Ki_s`).

---

## 4. Disturbance model (Stewart platform) — `simulation/stewart_platform.py`

Base angles and their analytic rates (the gyro's exogenous input):

```
β_y(t)  = e(t)·A_y·sin(ω_y t)              A_y = deg2rad(yaw_mag),  ω_y = 2π·f_y
β_p(t)  = e(t)·A_p·sin(ω_p t + π/2)        A_p = deg2rad(pitch_mag),ω_p = 2π·f_p
β̇_y(t)  = e(t)·A_y·ω_y·cos(ω_y t)
β̇_p(t)  = e(t)·A_p·ω_p·cos(ω_p t + π/2)
```

Ranges: magnitude 3–15°, frequency 0.1–0.4 Hz. `e(t) ∈ [0,1]` is a smooth
on/off envelope (ramp ~0.3 s) so toggling doesn't step the base angle.

On **disable**, the current base angle is *folded into the gimbal*
(`g += β`, then `e := 0`) so the LOS is continuous (`app.SimEngine.set_disturbance_enabled`).
**Check:** the analytic `β̇` omits the `ė·A·sin` envelope-derivative term — a
brief (~0.3 s) transient during ramp-up only; disable is exact (fold + e=0).

---

## 5. Line-of-sight kinematics — the key modelling step

Exact barrel direction and LOS angles (`visualization.barrel_world_direction`,
`los_angles`):

```
d      = R_base · R_gim · x̂ = Rz(β_y)Ry(−β_p) Rz(g_az)Ry(−g_el) · (1,0,0)ᵀ
LOS_az = atan2(d_y, d_x)
LOS_el = asin(d_z)
```

**Small-angle / single-axis-exact decomposition used for the control logic:**

```
LOS_az ≈ β_y + g_az        LOS_el ≈ β_p + g_el
```

This is **exact** when the other axis is zero and **first-order** otherwise
(cross-coupling between yaw and pitch is O(β·g), second order). Verify:
- `los_angles(g,0, β_y,0)` returns `az = β_y + g` exactly (tested).
- For combined yaw+pitch up to ~15°+15°, the exact `atan2/asin` differ from the
  additive model by a few tenths of a degree. **Auditor: confirm this bias is
  acceptable / doesn't accumulate.**

**LOS rate (gyro measurement):**

```
d/dt LOS_az ≈ ġ_az + β̇_y        d/dt LOS_el ≈ ġ_el + β̇_p
```

The gimbal part `ġ` is the exact plant state `turret.azimuth_rate`; the base
part `β̇` is the analytic rate from §4. The sum is the absolute LOS rate a rate
gyro on the barrel would report. This sum is formed in `app.SimEngine.advance`:
`los_rate = turret.<axis>_rate + β̇` and passed as the inner-loop feedback.

---

## 6. Controllers — `control/pi_controller.py`

Discrete PI with saturation and back-calculation anti-windup:

```
I      := I + e·dt
u      := Kp·e + Ki·I
u_clip := clip(u, [lo, hi])
if saturated and Ki≠0:  I := (u_clip − Kp·e)/Ki      # so Kp·e + Ki·I = u_clip
return u_clip
```

Continuous: `C(s) = Kp + Ki/s`. The outer position controller is a pure **P**
(`Ki = 0`), so it has no integrator and is not affected by the anti-windup branch.

---

## 7. Cascade architecture (per axis) — `control/control_system.py`

```
         φ_ref            +          ω_ref           +               u            plant            g (gimbal)
  ─────────────▶( Σ )────────▶[ Kp_pos ]────▶( Σ )────────▶[ Kp_s + Ki_s/s ]────▶[ 1/(τs+1) ]──┬──▶ ġ
                  ▲ −                           ▲ −                                             │   │
                  │                             │                                               │  (∫)
                  │  LOS_pos = β + g            │  LOS_rate = β̇ + ġ                             │   ▼
                  └───────────── (gyro position)┴──────────── (gyro rate) ◀───── β̇, β ◀────────┘   g
```

Signals and dimensions:

| symbol      | meaning                         | unit    |
|-------------|---------------------------------|---------|
| `φ_ref`     | commanded LOS angle             | rad     |
| `LOS_pos`   | measured LOS angle `β+g`        | rad     |
| `ω_ref`     | inner-loop speed reference       | rad/s   |
| `LOS_rate`  | measured LOS rate `β̇+ġ`         | rad/s   |
| `u`         | commanded gimbal rate → plant    | rad/s   |
| `Kp_pos`    | outer P gain                     | **1/s** |
| `Kp_s`      | inner P gain                     | — (dimensionless) |
| `Ki_s`      | inner I gain                     | **1/s** |

Defaults: `Kp_pos = 8`, `Kp_s = 6`, `Ki_s = 2`, `τ = 0.2 s`, `u_max = ω_ref,max = 180°/s`.

**Dimensional check of the outer gain:** `ω_ref = Kp_pos·(φ_ref − LOS_pos)` →
`[rad/s] = [1/s]·[rad]`. ✔. Because both the error and the produced speed
reference scale with the same angular unit, `Kp_pos` is numerically identical
whether you think in rad or deg — that's why the slider (1–20) is unit-agnostic.

**Inner-loop dimensional check:** `u = Kp_s·(ω_ref−LOS_rate) + Ki_s·∫(…)` →
`[rad/s] = [–]·[rad/s] + [1/s]·[rad/s·s]`. ✔.

---

## 8. The three modes — reference source & error definition

`ControlSystem.step(t, LOS_az, LOS_rate_az, LOS_el, LOS_rate_el, target_az, target_el)`.

**Mode 1 — SPEED** (inner loop only; the outer P is bypassed):
```
ω_ref_az = r(t)              # speed reference (deg/s → rad/s)
ω_ref_el = 0                 # hold: drive LOS_el rate to 0  → elevation stabilised
u = PI(ω_ref − LOS_rate)     # feedback is the ABSOLUTE LOS rate
graphed error = ω_ref − LOS_rate      (a rate; deg/s)
```
Optional toggle `speed_shows_target_error`: reports `target − LOS_pos` (a
position error, deg) for the graph/log instead — a readout only; `u` unchanged.

**Mode 2 — POSITION** (full cascade):
```
φ_ref_az = r(t)              # position reference (deg → rad)
φ_ref_el = el_hold           # = board elevation (points at the board vertically)
ω_ref = Kp_pos·(φ_ref − LOS_pos);  u = PI(ω_ref − LOS_rate)
graphed error = φ_ref − LOS_pos       (a position; deg)
```

**Mode 3 — TARGET** (full cascade; auto-aim):
```
φ_ref = (target_az, target_el)   # geometric angles to the board centre from the pivot
… same cascade …
graphed error = target − LOS_pos = the barrel/target pointing error
```

`error_is_rate = (mode == SPEED and not speed_shows_target_error)` drives the
graph/CSV units (deg/s vs deg).

Reference signals `r(t)`: square / sine / **constant** (DC setpoint) — `control/reference_signals.py`.

---

## 9. The central correctness argument: why LOS-rate (gyro) feedback

This is the claim most worth checking.

**Symptom before the fix:** the inner loop used the *gimbal-relative* rate `ġ`
(`turret.azimuth_rate`) as feedback, not `β̇+ġ`. Then in SPEED mode the base rate
`β̇` never entered any error, so a zero speed command drove `ġ→0` while the base
kept moving — the LOS wobbled the full `±A` (no rejection). POSITION/TARGET
appeared stabilised only because the *outer* position loop (on `LOS_pos = β+g`)
saw and corrected the disturbance; the inner loop still never saw it.

**After the fix:** feedback is the absolute LOS rate `LOS_rate = β̇ + ġ`.
- SPEED, `ω_ref = 0`: error `= 0 − (β̇+ġ)`; the PI drives `ġ → −β̇`, so
  `LOS_rate → 0` and the LOS **holds** against base motion — genuine
  stabilisation with no outer loop.
- Non-zero `ω_ref`: the loop makes `LOS_rate → ω_ref`, i.e. the *absolute*
  pointing slews at the commanded rate regardless of base motion (gyro slew).

**Boundedness in SPEED mode (no position loop):** `LOS_pos = ∫ LOS_rate`. The
residual LOS rate (imperfect rejection at the disturbance frequency) is a
zero-mean sinusoid, so its integral is a *bounded* sinusoid — the LOS position
oscillates with small amplitude and does **not** drift. Empirically: a 10° /
0.2 Hz base wobble with `ω_ref=0` gives LOS peak ≈ 1.6° (vs ~10° open-loop).
**Auditor: confirm no secular drift and that the residual scales ~ 1/loop-gain.**

**Stability claim:** adding `β̇` to the feedback is injecting an *exogenous*
signal into the summing junction; it does not alter the loop's characteristic
equation (the transfer from `u` to the measured `ġ` is unchanged — `β̇` is not a
function of `u`). Hence it changes the *disturbance response* but not closed-loop
stability. **Auditor: verify this — it's the crux of why the change is safe.**

---

## 10. Approximations & assumptions to audit (checklist)

1. **Additive LOS model** (`LOS ≈ β + g`, `LOS_rate ≈ β̇ + ġ`) vs the exact
   rotation composition used for `LOS_pos`. Exact on a single axis; O(β·g)
   cross-coupling otherwise. Confirm the bias is small for ≤15°+15° and does not
   accumulate into a steady pointing offset.
2. **Analytic base rate** ignores the envelope derivative `ė` (transient only,
   ~0.3 s on enable). Disable is exact (fold + `e=0`).
3. **Outer loop is P-only** ⇒ type-0 in position ⇒ *finite* steady-state error
   to a constant reference and a bounded lag/error to a sinusoid
   (≈ magnitude / loop gain). Deliberate. Confirm this is acceptable, or note an
   outer integrator would zero constant-reference error.
4. **Discrete-time**: `dt = 0.01 s`, 3 substeps/frame, semi-implicit Euler.
   Nyquist ≫ all dynamics (`τ=0.2 s`, disturbance ≤0.4 Hz, gains chosen for
   bandwidth ~a few Hz). Confirm no discretization instability at max gains
   (`Kp_pos=20`, `Kp_s=20`, `Ki_s=20`).
5. **Elevation reference in POSITION mode** is fixed at the board elevation
   (`el_hold`), not user-driven — a scoping choice, not a control result.
6. **Anti-windup** by back-calculation on saturation, on the inner PI only
   (outer is P). Confirm the cascade cannot wind up (both stages clamp; the
   outer P has no state).
7. **`Kp_pos` unit-agnosticism** (rad vs deg give the same number) — verify §7.
8. **Sign of `Ry(−el)`** gives positive-up elevation — verify §2.
9. **Plant already contains a rate servo** (`τ` lag); the inner PI is thus a
   rate loop around a rate servo. Confirm the doubled rate dynamics don't cause
   an unexpected resonance at high `Ki_s`.

---

## 11. Where to look in code

| Concept | File / symbol |
|---|---|
| Plant (velocity servo) | `simulation/turret_model.py` (`_advance_axis`, `step`) |
| PI + anti-windup | `control/pi_controller.py` (`step`) |
| Cascade + modes | `control/control_system.py` (`ControlSystem.step`) |
| Reference signals | `control/reference_signals.py` |
| Disturbance + rates | `simulation/stewart_platform.py` (`yaw/pitch`, `yaw_rate/pitch_rate`) |
| LOS composition/rate wiring | `simulation/visualization.py` (`los_angles`, `barrel_world_direction`); `app.SimEngine.advance` |
| Disable-fold (LOS continuity) | `app.SimEngine.set_disturbance_enabled` |

---

## 12. Concrete properties an auditor can test

- SPEED, `ω_ref=0`, disturbance on ⇒ LOS peak ≪ base amplitude, no drift.
- SPEED, constant `ω_ref`, disturbance off ⇒ gimbal (and LOS) rate → `ω_ref`.
- POSITION, constant `φ_ref` ⇒ `LOS_pos → φ_ref` (small residual under disturbance).
- TARGET ⇒ `LOS_pos → (target_az, target_el)`; barrel/target error → ~0.
- Toggling the disturbance leaves `LOS_pos` continuous (no step).
- `yaw_rate(t)` equals `d/dt yaw(t)` numerically.
- Dimensional consistency of every summing junction in §7–§8.
- Disturbance rejection improves (or is unchanged) vs gimbal-rate feedback, and
  closed-loop stability is preserved (§9).

These are exercised in `tests/test_system.py` (run: `uv run python tests/test_system.py`).
