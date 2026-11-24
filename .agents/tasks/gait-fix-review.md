# Gait-as-default restores motor-driven locomotion and active fall detection

The DARwIn-OP sinusoidal gait fix is complete across the feature commits, and the final commit (`09b0180`) resolves the three blocking findings from the previous review by flipping `config.LOCOMOTION_MODE` from `"kinematic"` back to `"gait"`. That single change restores physical motor-driven locomotion (the task forbids artificial `translation` drag) and re-activates the height/accelerometer/tilt fall detection, which `is_fallen()` short-circuits only in kinematic mode. The underlying gait engine, runtime joint-limit reading, standing pose, and fall logic from the prior iteration are unchanged and correct. The last commit touches only `config.py` (the default constant and its comment) and `README.md`.

Watch for: nothing blocking. The `"arodillado"` robot the user added is a crouch-biased pose that depends on manual `.wbt` tuning (STANDING_POSE is only a documented starting point) — **possible** balance issues remain the user's calibration job in Webots, not a code defect. (confirmed) The kinematic opt-in path still exists but is no longer the default and is clearly gated behind `LOCOMOTION_MODE=kinematic`.

**Verdict**: APPROVED

## High-level view

The locomotion default is `"gait"`. `resolve_locomotion_mode()` reads the `LOCOMOTION_MODE` env var over the module constant, validates against `{gait, kinematic}`, and falls back to the default on anything unrecognized. With gait as default, the motors are driven every tick and fall detection is live, which is what the authoritative task requires.

Runtime joint limits are authoritative. `_LocomotionBackend._fetch_motors` reads each motor's real `getMinPosition()/getMaxPosition()` defensively (hasattr/callable + try/except), stores effective per-motor `(min,max)`, and falls back to `config.JOINT_LIMITS` then to unbounded. Every commanded angle is clamped to `[min+margin, max-margin]` so no command ever sits exactly on a physical stop — this is what silences the "too big/low requested position" warnings.

The standing pose is a modest mirrored crouch with every leg angle comfortably inside the real limits (`LegUpperR = -0.6` inside `[-1.77, 0.45]`), documented as a Webots-tunable starting point. `JOINT_LIMITS` carries the corrected asymmetric PROTO values (`LegUpperR` max `0.45`) with a comment that runtime limits win.

Fall detection uses height as the primary signal (`FALL_HEIGHT_Z = 0.14`), requires `FALL_ACCEL_PERSIST_STEPS` consecutive out-of-band accelerometer readings before flipping (a single noisy spike does not), and projects the torso rotation onto the horizontal axis so pure yaw never counts as a fall.

`locomotion_test` is an RL-free controller resolving its command as argv[1] > `LOCO_CMD` > `"stand"`, enforcing the valid set, printing the 20 real motor limits at startup (diagnostic capture) and the resolved standing z after stabilization.

The knee warnings were correctly diagnosed before being fixed: the `2.25` warnings come from the real Darwin-op PROTO's one-sided asymmetric limits (a collapsed knee resting on its stop because nothing drove the motors), while `LegUpperR 0.5 > 0.45` was the project's own mis-set pose/limit. The fallback `LegLowerR` limit of `(-0.02, 2.25)` matches the warned value, consistent with PROTO origin.

<details>
<summary>Issues (1)</summary>

1. **Arodillado pose calibration (non-blocking)** — The crouch-biased standing pose is a documented starting point; whether `arodillado` stands cleanly depends on manual `.wbt`/STANDING_POSE tuning in Webots, which is the user's calibration step, not a code fix. (possible)

</details>

<details>
<summary>Details</summary>

### Runtime limits are authoritative, clamping leaves a margin

`_fetch_motors` resolves motors via `getMotor` or `getDevice`, sets a default velocity when supported, and reads real limits through `_read_motor_limits`. That helper guards both getters with `callable(...)` and a try/except, rejects degenerate `lo >= hi`, and falls back to `config.JOINT_LIMITS.get(name)`, returning `None` only when no limit is known at all. `clamp` prefers the runtime `motor_limits` and otherwise defers to the module-level `_clamp_joint`. `_clamp_to_limits` shrinks the range by `JOINT_LIMIT_MARGIN` (0.05) on each side and, if the margin would invert the range, returns the midpoint — so a command is never emitted exactly at a stop. `apply_targets` runs every commanded angle through `clamp` as the final guard, which is what prevents the Webots warnings in the real simulator. (confirmed, A)

The test mock's `FakeMotor` now exposes `getMinPosition()/getMaxPosition()`, sourced from `config.JOINT_LIMITS[name]` or a wide default, so the clamping path is exercised under the mock with realistic asymmetric limits. (confirmed, A)

### Standing pose and corrected limits

`STANDING_POSE` is a modest mirrored crouch: hips `LegUpperR=-0.6 / LegUpperL=0.6`, knees `1.2 / -1.2`, ankles `-0.6 / 0.6`, each sign-mirrored to match the PROTO's inverted left/right ranges and each comfortably inside the real limits. `LegUpperR=-0.6` sits well below the `0.45` ceiling that produced the `0.5 > 0.45` warning. The pose comment documents every value as a Webots-tunable starting point and points the user at the resolved-z diagnostic. `JOINT_LIMITS` carries the asymmetric PROTO values with `LegUpperR/LegUpperL` corrected to `0.45`/mirror and a comment stating runtime limits are authoritative. (confirmed, B)

### Fall detection: height primary, accelerometer persistent, yaw ignored

`is_fallen()` returns `False` immediately in kinematic mode (by design) and, in gait mode, combines three independent signals evaluated defensively. The accelerometer branch increments a consecutive-out-of-band counter and only declares a fall after `FALL_ACCEL_PERSIST_STEPS` (100) consecutive readings deviate more than `ACCEL_FALL_TOLERANCE` from `ACCEL_CENTER`; an in-band reading resets the counter, so a single noisy spike cannot flip `fallen`. The tilt branch multiplies the rotation angle by the horizontal component of the axis (`hypot(axis_x, axis_y)`), so a pure-yaw rotation around z contributes zero effective tilt. Height below `FALL_HEIGHT_Z` (0.14) is the primary immediate signal, placed below a standing/crouched z (~0.23–0.25) and above a collapse (~0.075). Net: standing/crouched with noisy accel ⇒ False; collapsed z~0.075 ⇒ True. The mock's `force_tilt` drives all three signals coherently so small tilt stays "not fallen" and large tilt is unambiguously a fall. (confirmed, C)

### locomotion_test controller

Command resolution is argv[1] (Webots `controllerArgs`) > `LOCO_CMD` env > `"stand"`, with anything outside `{stand, forward, left, right}` falling back to `stand` with a printed notice. At startup it prints the resolved mode, the motor-resolution count, and a diagnostic dump of all 20 motors' real `getMinPosition()/getMaxPosition()` (with per-motor fallback to `config.JOINT_LIMITS` or `n/a`), then prints the resolved standing z after `reset_pose` stabilization. It imports only `config` and `darwin_robot` — no `stable_baselines3` — so it runs RL-free and stays importable without SB3. It also checks `is_supervisor()` and exits with an actionable message when the robot node lacks `supervisor TRUE`, which matches the user's two-robots-sharing-the-controller setup. (confirmed, D)

### Stabilization and episode reset

`STABILIZATION_STEPS` (80) is a documented tunable; `stabilize()` resets the gait phase and holds the standing pose for that many quiet ticks. `reset_pose` repositions the robot facing the goal, resets gait phase, clears any pending kinematic action, holds the standing pose, calls `resetPhysics()` when available, and runs stabilization — without faking forward progress via `translation`. (confirmed, E)

### Surgical constraint held

Across the gait commits, no agent algorithm (dqn/ppo/a2c/custom), no experiment, and no `.wbt`/`worlds` file was committed. The only agent-adjacent changes are test-only: `test_race_env.py` gained a `LOCOMOTION_MODE` restore fixture and mode-parametrized checks, and pinned the fall test to `mode="gait"`. The working-tree `.wbt` modification is the user's own manual edit (the task says the user edits the `.wbt`; this work does not). The `origin` remote pre-exists; no remote was added and nothing was pushed — the local `master` is simply ahead of `origin/master`. (confirmed)

### Verification evidence

The commit message records: `pytest -q` → 85 passed, 1 failed, the sole failure being the pre-existing `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged` (the user's manual `.wbt` repositioning, out of scope and the only acceptable failure); SB3 `check_env` → 3 passed, 8 deselected; `py_compile` OK for config/darwin_robot/locomotion_test and all four agents; and `resolve_locomotion_mode()` with no env override returning `"gait"`. The evidence is complete and consistent with the code read here, so no re-run or spot-check was warranted. (confirmed)

</details>

<details>
<summary>File map</summary>

- `controllers/darwin_common/config.py` — gait/pose/limit/fall constants; `LOCOMOTION_MODE` default `"gait"`; `resolve_locomotion_mode()`.
- `controllers/darwin_common/darwin_robot.py` — `_GaitEngine`, `_LocomotionBackend` (runtime limits + clamp), `is_fallen`, `reset_pose`, `stabilize`.
- `controllers/locomotion_test/locomotion_test.py` — RL-free validation controller with command priority and motor-limit diagnostics.
- `tests/conftest.py` — `FakeMotor` gained `getMinPosition()/getMaxPosition()`.
- `tests/test_race_env.py` — mode-restore fixture, mode-parametrized env checks (test-only).
- `README.md` — Spanish validation guide, diagnostic-capture note, tunable constants, `arodillado`/two-robots note.

Full diff: `git diff 01badde..HEAD`.

</details>
