# Python sinusoidal gait replaces the DARwIn-OP locomotion stub

The change replaces a non-functional locomotion stub (which only called the test-mock-only `queue_action` and never drove any motor) with a Python CPG-style sinusoidal gait that commands the 20 DARwIn-OP motors directly via `setPosition` every tick. All tunables move into `config.py`; `is_fallen()` is rewritten to combine the centered-~512 accelerometer model with Supervisor orientation and torso height; `reset_pose()` now resets gait phase, calls `resetPhysics`, and runs a stabilization period; and a standalone RL-free `locomotion_test` controller is added for single-robot validation in Webots. DQN/PPO/A2C/custom algorithm logic is untouched — agents reach the new gait only through `DarwinRaceEnv`. The commit touches neither world file nor `requirements.md`.

Watch for: the turn gait is a static hip-yaw offset with no stepping oscillation during turns, so in-place turning stability is unverified and is a Webots tuning risk (likely). Non-knee joint limits are documented assumptions, not read from the PROTO (confirmed, author-acknowledged). Gait quality overall is unvalidated against a real runtime by design — the sandbox has no Webots (confirmed).

**Verdict**: APPROVED

## High-level view

Locomotion is centralized in `darwin_robot.py` through a new `_GaitEngine` holding the gait phase and two command amplitudes (x for forward, a for turn), analogous to the C++ `setXAmplitude`/`setAAmplitude`. `_LocomotionBackend` fetches the 20 motors via `getMotor`/`getDevice` and applies `setPosition` to the engine's per-joint targets each tick inside `DarwinRobot.tick()`. When no motor resolves (the headless mock), it falls to a documented no-op and the existing `queue_action` kinematics still advance the robot, which is what keeps the headless suite meaningful.

Every tunable lives in `config.py`: motor names, standing pose, period/frequency, per-joint amplitudes, lateral sway, leg phase offset, joint limits, safety margin, motor velocity, stabilization steps, and the three fall thresholds. Each constant carries a one-line comment and an "AJUSTABLE" marker for the user to tune in Webots.

Joint safety is enforced by a single `_clamp_joint` helper applied to both the standing pose and every computed gait target, clamping to `[min+margin, max-margin]`. The knee range is fixed at `[-2.25, 2.25]`; a dedicated test sweeps a full gait cycle and asserts no commanded angle leaves the safe band.

Fall detection now fires on any of three independent signals — accelerometer deviation from ~512 beyond tolerance, torso tilt from the Supervisor `rotation` field, or torso height below threshold — each guarded so a missing sensor or field skips that branch instead of raising. The mock's `force_tilt` was rewritten into a single coherent "force a fall" switch that moves all three signals together.

The known gaps are turn-gait quality, the assumed non-knee joint limits, and the fact that physical walking quality can only be validated by the user in Webots. The knee-warning origin (uncontrolled collapse resting against the `LegLowerR/L` limit, not code commanding ±2.25) is confirmed by code reasoning and documented; the final visual confirmation is deferred to the user's Webots run.

<details>
<summary>Issues (3)</summary>

1. **Turn gait has no stepping** — `turn_left`/`turn_right` only apply a static `PelvYR/PelvYL += a*0.3` yaw offset with no leg oscillation, so a stationary turn may not produce stable rotation. Validate turning in Webots and add leg stepping during turns if the robot does not rotate.
2. **Non-knee joint limits are assumed** — `JOINT_LIMITS` for every joint except the knee uses conservative guesses because the PROTO was unreachable in the sandbox (author-documented). Reconcile against `Darwin-op.proto` when network access exists, in case the real ranges are tighter and cause excessive clamping.
3. **Physical gait quality unvalidated** — no Webots runtime in the sandbox; stand/forward/turn stability and the cessation of knee warnings are unconfirmed. Run `locomotion_test` with each `LOCO_CMD` before training the four agents.

</details>

<details>
<summary>Details</summary>

### Gait engine and direct motor control

`_GaitEngine` builds each tick's targets from `STANDING_POSE` plus coordinated sinusoids: hips (`LegUpper`) and ankles track `sin(phase)`, knees (`LegLower`) use `abs(sin(phase))` so they always flex rather than extend during swing, and a `cos(phase)` term sways pelvis and ankles laterally to shift weight. Left and right legs are driven `GAIT_LEG_PHASE_OFFSET = math.pi` apart. The forward amplitude `x` scales the stride; the turn amplitude `a` adds a hip-yaw asymmetry. `DarwinRobot.tick()` calls `backend.update()` (advance phase, compute targets, `setPosition` on every resolved motor) before stepping the backend, so the gait is recomputed every simulation step rather than set once per action — this is the core behavioral fix over the stub.

The real-vs-mock split is the load-bearing design decision. In Webots the motors resolve and the gait moves the legs; in the headless mock no motors resolve, so `available` is false, a `RuntimeWarning` is emitted, and the robot still advances through the preserved `_queue(...)` → `_apply_kinematics` path. This is what lets the suite keep testing env/agent behavior without a runtime while the `setPosition` commands remain the only thing that matters inside Webots. The old `test_locomotion_fallback_triggers_on_managers_failure` was replaced by `test_locomotion_noop_when_no_motors_resolve`, which nulls `getMotor`/`getDevice` and asserts the no-op path warns and does not raise.

### Joint clamping with margin

```python
def _clamp_joint(name, angle):
    limits = config.JOINT_LIMITS.get(name)
    if limits is None:
        return float(angle)
    lo, hi = limits
    margin = config.JOINT_LIMIT_MARGIN
    lo_safe, hi_safe = lo + margin, hi - margin
    if lo_safe > hi_safe:
        return float((lo + hi) / 2.0)
    return float(max(lo_safe, min(hi_safe, angle)))
```

Both `standing_pose()` and `compute_targets()` route every angle through this helper, so no joint is ever commanded to its exact physical limit. `test_gait_targets_stay_within_safe_limits` sweeps 60 phase samples across a full cycle at the configured x/a amplitudes and asserts each target stays within `[min+margin, max-margin]` for every joint, which directly addresses the knee `[-2.25, 2.25]` requirement and the source of the Webots "too big/low requested position" warnings.

### Reset, stabilization, and no translation faking

`reset_pose()` repositions to the start (legitimate teleport), sets heading to 0, resets the gait phase, commands the standing pose, calls `resetPhysics()` on the self node when present (guarded so the mock is unaffected), then runs `stabilize()` — `STABILIZATION_STEPS` quiet standing ticks. `race_env.reset()` was not modified: it already calls `reset_pose` before capturing `_prev_dist`, so stabilization runs before the baseline distance is read, which is the behavior the plan required. There is no `translation` mutation inside `step()`; forward progress must emerge from the gait (in Webots) or from the mock kinematics (headless).

### Three-signal fall detection

`is_fallen()` returns true on the first of: accelerometer max-axis deviation from `ACCEL_CENTER` (512) exceeding `ACCEL_FALL_TOLERANCE` (80, matching the Walk.cpp band); torso tilt from the Supervisor `rotation` field — projected onto the horizontal axis so a pure yaw does not count as a fall — exceeding `FALL_TILT_THRESHOLD`; or torso z below `FALL_HEIGHT_Z` (0.18). Each branch is wrapped so a missing sensor or field skips it. The mock default accelerometer changed from `(0,0,9.81)` to `(512,512,512)`, and `force_tilt` now moves all three signals coherently (accel deviation, a rotation about the x axis, and a lowered z for large tilts, restored z for small ones). All four robot start z values (0.24 / 0.34) sit above `FALL_HEIGHT_Z`, so a standing robot is never falsely flagged. New `test_is_fallen_true_on_low_height` and `test_is_fallen_true_on_strong_tilt` cover the height and tilt branches; the unchanged `test_race_env.py::test_fall_penalizes_and_terminates` still passes because `force_tilt(threshold+0.5)` drives both accel and height into the fallen region.

### Turn gait limitation

`turn_left`/`turn_right` set the engine's `a` amplitude, but `compute_targets` only translates `a` into a constant `PelvYR/PelvYL += a*0.3` yaw bias — there is no phase-dependent leg stepping tied to `a`. Forward walking has full hip/knee/ankle oscillation; turning does not. A static yaw offset against planted feet may rotate the torso without taking steps, which can stall or destabilize an in-place turn. This cannot be exercised headless (the mock advances via `queue_action`, not the gait), so it is a Webots-only validation item and a likely tuning gap rather than a confirmed bug.

### Standalone locomotion_test controller

`controllers/locomotion_test/locomotion_test.py` instantiates a `Supervisor`, builds a `DarwinRobot`, reports resolved-motor count against the 20 names, reads `LOCO_CMD` (stand/forward/left/right, default stand), runs `reset_pose` for the initial stabilization, then loops applying the chosen high-level action and `tick()` while logging position, heading, height, accelerometer, and fall state every 10 steps. It imports no SB3 or RL code and drives the Task-2 `_GaitEngine` end to end. Its module docstring records the knee-warning finding, satisfying the Task-1 empirical-confirmation requirement at the code level (final visual confirmation is the user's Webots run).

### Verification evidence

The commit message and `.agents/tasks/VERIFICATION.md` record `pytest -q` → 58 passed, 1 failed (the pre-existing `test_wbt` geometry mismatch caused by the user's manual world edits, out of scope and explicitly forbidden to fix), `check_env` passing via `test_check_env_sb3`, and `py_compile` clean on the new/touched controllers. The 50→58 jump matches the 8 new gait tests. The evidence is internally consistent with the diff — standing z values clear the fall-height threshold, `force_tilt` math clears the accel tolerance, and the no-motor path is tested — so no suite was re-run. Public signatures (`walk_forward`/`turn_left`/`turn_right`/`stand`/`tick`/`get_position`/`get_orientation`/`is_fallen`/`reset_pose`, plus `find_others`/`find_barriers`) are all preserved.

</details>

<details>
<summary>File map</summary>

- `controllers/darwin_common/config.py` — all gait tunables added (motor names, standing pose, period/frequency, amplitudes, sway, phase offset, joint limits + margin, motor velocity, stabilization steps, fall thresholds).
- `controllers/darwin_common/darwin_robot.py` — `_GaitEngine`, `_clamp_joint`, rewritten `_LocomotionBackend` (direct `setPosition`), gait-driven `tick()`, `stabilize()`, rewritten `is_fallen()` and `reset_pose()`.
- `controllers/locomotion_test/locomotion_test.py` — new RL-free single-robot validation controller (`LOCO_CMD`).
- `tests/conftest.py` — accelerometer default centered ~512; `force_tilt` rewritten as a coherent 3-signal fall switch.
- `tests/test_darwin_robot.py` — new gait tests (clamp, safe-limit sweep, phase advance, 180° phase, nominal pose, amplitude mapping, two fall branches); fallback test replaced with no-motor no-op test.
- `README.md` — "Locomoción (gait sinusoidal)" section: tunables, `locomotion_test`/`LOCO_CMD` usage, knee-warning origin, validate-then-train flow.
- `.agents/tasks/VERIFICATION.md`, `.agents/tasks/gait-plan.md` — task artifacts.

Full diff: `git show 2ba266d`.

</details>
