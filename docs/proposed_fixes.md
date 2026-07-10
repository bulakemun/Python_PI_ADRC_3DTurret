# Proposed fixes — control-logic audit (2026-07-10)

Findings from an audit of the control theory, code, and docs. Everything was
verified against the running simulation (all 49 system checks pass; numeric
experiments below were run against `SimEngine` directly). Ordered by priority.
Nothing here is applied yet — this is a proposal.

---

## 1. Gyro model misses the envelope-derivative term (real behavioral bug)

**Where:** `simulation/stewart_platform.py` — `yaw_rate()` / `pitch_rate()`.

**Problem:** The base angle is `β(t) = e(t)·A·sin(ωt)` (with the on/off
envelope `e`), so the true rate is

```
β̇ = ė·A·sin(ωt) + e·A·ω·cos(ωt)
```

but the code reports only the second term. During the 0.3 s enable ramp,
`ė ≈ 1/0.3 ≈ 3.3 s⁻¹`, so the gyro can under-report by tens of deg/s.
The control loop never "sees" the base angle sweep in.

**Measured impact:** enabling a 15° / 0.2 Hz yaw disturbance while
`sin(ωt) ≈ 1`, in SPEED mode with `ω_ref = 0`, leaves a **+14.7° permanent LOS
offset** (there is no position loop in SPEED mode to pull it back).
POSITION/TARGET modes recover; SPEED does not.

`docs/control_theory.md` §4 calls this "a brief (~0.3 s) transient" — that
understates it badly and should be corrected too (see item 5).

**Proposed fix (preferred):** track the envelope slope in
`StewartDisturbance.advance()` (it is known exactly: `±1/ramp_time` while
ramping, else 0) and add the `ė·A·sin(...)` term to `yaw_rate()` /
`pitch_rate()`. This makes the gyro model consistent with the angle model at
all times; no controller changes needed.

**Alternative:** on enable, restart the sinusoid at a zero crossing (store a
time offset so `sin` starts at 0). Simpler math, but changes the disturbance
phase behavior and the rates are still slightly wrong during the ramp.

**Regression test:** enable the disturbance at a sine peak in SPEED mode with
`ω_ref = 0`; assert the post-settle mean LOS offset is < ~1° (currently ~14.7°).

---

## 2. Doc claim "disable is exact (fold + e=0)" is false

**Where:** `docs/control_theory.md` §4 and checklist item 2; behavior in
`app.SimEngine.set_disturbance_enabled`.

**Problem:** the disable-fold `g += β` is additive, but the true LOS is the
rotation composition `Rz(β_y)Ry(−β_p)·Rz(g_az)Ry(−g_el)`. The fold is exact
only to first order; with both axes nonzero there is an O(β·g) LOS jump.

**Measured impact:** base 15°/15°, gimbal 10°/5° → **0.59° azimuth jump** on
disable. (The existing test asserts < 0.5° only in a milder case.)

**Proposed fix:** doc-only — change "disable is exact" to "exact to first
order; worst-case ~0.6° jump at 15°+15° base with a deflected gimbal". If an
exact fold is ever wanted, solve `los_angles(g', 0) = los_angles(g, β)` for
`g'` (closed form: `g'_az = LOS_az`, `g'_el = LOS_el` from the exact
composition) instead of the additive fold.

---

## 3. Slider ranges allow a dead or near-unstable loop

**Where:** `app._make_control_panel` — inner-gain sliders both floor at 0.

**Problem/measured:** `Kp_s = 0, Ki_s = 0` is fully open loop (turret dead);
`Kp_s = 0, Ki_s = 0.5` with `Kp_pos = 20` gives a ~73° peak oscillation (the
outer loop demands bandwidth the gutted inner loop can't deliver).

**Proposed fix (pick one):**
- floor the inner sliders above zero (e.g. `Kp_s ≥ 0.5` or `Ki_s ≥ 0.5`), or
- keep the ranges as a deliberate "let the user destabilize it" teaching
  feature, but say so in CLAUDE.md / control_theory.md so it isn't mistaken
  for a bug later.

---

## 4. Architecture diagram omits the inner-loop (gyro rate) feedback

**Where:** `docs/architecture.svg`.

**Problem:** only the LOS *position* feedback path into the outer summing
junction is drawn. The inner summing junction has no feedback arrow — the
absolute LOS-*rate* (gyro) feedback, the centerpiece of commit `5d51141`, is
missing from the diagram.

**Proposed fix:** add a second feedback path from the LOS/plant side into the
inner sum, labelled "LOS rate (gyro: ġ + β̇)". While there, the disturbance
box could also show its `β̇` output feeding that path.

---

## 5. Stale doc numbers and understated caveats

- CLAUDE.md ("Tests:" paragraph) says **25 checks**; the suite currently runs
  **49**. Suggest wording that doesn't need manual upkeep ("~50 checks" or
  drop the count).
- `docs/control_theory.md` §4 and checklist item 2: rewrite per items 1 and 2
  above (envelope-derivative impact is *not* a brief transient in SPEED mode;
  disable-fold is first-order, not exact).

---

## 6. Minor / latent (no action needed now, listed for the record)

- **Azimuth error doesn't wrap at ±π** (`control_system.py`, position error).
  Unreachable with current ≤90° references, but will bite when targets can be
  behind the turret. Fix when scope grows: wrap the error into (−π, π].
- **Outer P controllers accumulate an unused integral** (`ki=0` skips
  anti-windup, `_integral` grows forever). Harmless today; if an outer Ki
  slider is ever added, reset or stop accumulating when `ki == 0`.
- **Anti-windup is blind to the elevation hard stop**: the plant zeroes the
  rate at ±90° (`turret_model.step`), which the inner PI can't see, so its
  integrator can wind at the stop. Only reachable near the limits.
- **TARGET angles use the rest pivot** while the real pivot rides the base
  tilt — ~0.03° error at 400 m. Negligible at current scope.
- **`stewart._env` is poked directly** from `app.py` and tests. A small
  `snap_to(enabled: bool)` method on `StewartDisturbance` would keep the
  envelope private.

---

## What was verified as correct (no action)

- Cascade structure, dimensional bookkeeping, and Kp_pos unit-agnosticism
  match `docs/control_theory.md` §7 exactly.
- §9 stability argument (β̇ is exogenous; feedback change alters disturbance
  response, not the characteristic equation) is correct. Empirically: no LOS
  drift over 180 s (mean shift 1e-5 deg), residual halves when inner gains
  double (1.44° → 0.77°).
- Max-gain discrete stability (Kp_pos=Kp_s=Ki_s=20, 15°/0.4 Hz): LOS peak
  0.27°, no blow-up.
- Additive LOS-rate approximation: worst-case ~3.1 deg/s error at 15°/0.4 Hz
  on both axes; absorbed as extra disturbance, doesn't accumulate.
- Sign conventions (`Ry(−el)` = positive-up), plant model, PI anti-windup,
  and semi-implicit Euler integration all match the docs.
