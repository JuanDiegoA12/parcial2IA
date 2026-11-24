# Implementation Plan — Fix & tune the Python sinusoidal gait of the DARwIn-OP

Surgical fixes to the shared locomotion layer so the robot stands balanced, stays
within the real per-motor limits, and no longer registers as fallen while
standing. NO Webots runtime in this sandbox: verify with the project venv's
pytest suite, stable-baselines3 `check_env`, and `py_compile`. Physical gait
validation is the user's job in Webots, documented in the plan.

## Scope and ground rules (read before implementing)

- Edit ONLY: `controllers/darwin_common/config.py`,
  `controllers/darwin_common/darwin_robot.py`,
  `controllers/locomotion_test/locomotion_test.py`, `tests/conftest.py`,
  `tests/test_darwin_robot.py`, and `README.md` (doc updates). Do NOT touch the
  agent algorithms (`dqn_agent`, `ppo_agent`, `a2c_agent`, `custom_agent`),
  `race_env.py`, `agent_runner.py`, `experiments/`, or any `.wbt`/geometry.
- Run tests with the project venv ONLY (system python3 is PEP668-managed):
  `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q`
- Commit locally on branch `master`. The repo has NO remote: never add a remote
  and never push.
- Baseline before any change: `58 passed, 1 failed`. The ONE pre-existing
  failure is `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`
  (the user repositioned robots inside the `.wbt`, so the DQN translation is
  `(0.21881, 9.95112, 0.22402)` not `(0.11, 9.95, 0.24)`). It is unrelated to
  locomotion and OUT OF SCOPE (fixing it requires editing the `.wbt`). Keep it
  the only failure; every other test must stay green.

## Investigation finding — origin of the knee / hip warnings (record this)

The warnings come from the REAL Darwin-OP model, not the mock. The world uses
the remote PROTO declared at the top of `pista_atletismo_50m.wbt`:
`EXTERNPROTO ".../R2025a/projects/robots/robotis/darwin-op/protos/Darwin-op.proto"`.
That PROTO defines ASYMMETRIC, sign-mirrored limits per joint; the hardcoded
`config.JOINT_LIMITS` table (symmetric `(-1.57,1.57)`, knees `(-2.25,2.25)`) is
wrong. Authoritative values read from the R2025a Darwin-op PROTO
([Darwin-op.proto](https://github.com/cyberbotics/webots/blob/R2025a/projects/robots/robotis/darwin-op/protos/Darwin-op.proto),
content rephrased for compliance with licensing restrictions):

| Motor      | min    | max   | Notes |
|------------|--------|-------|-------|
| LegUpperR  | -1.77  | 0.45  | confirmed; matches the user's `0.5 > 0.45` warning |
| LegUpperL  | -0.50  | 1.68  | mirror of R |
| LegLowerR  | -0.03  | 2.25  | mirror; explains `too big 2.25` on R |
| LegLowerL  | -2.25  | 0.03  | explains `too low -2.25` on L |
| AnkleR     | -1.22  | 1.39  | mirror |
| AnkleL     | -1.39  | 1.22  | |
| PelvR      | -1.01  | 1.01  | |
| PelvL      | -1.00  | 0.93  | |
| PelvYR     | -2.42  | 0.66  | |
| PelvYL     | -0.69  | 2.50  | |
| Neck       | -1.81  | 1.81  | |
| Head       | -0.36  | 0.94  | |
| ArmUpperL  | -2.25  | 0.77  | ArmUpperR is the sign-flipped mirror |
| ArmLowerL  | -1.18  | 1.63  | ArmLowerR is the sign-flipped mirror |

Confirmed cause chain: the previous stub only called `supervisor.queue_action`
(absent in real Webots), so nothing drove the motors, the robot collapsed, and
the knees came to rest against their one-sided limits (`2.25` on R, `-2.25` on
L) — hence `too big/low requested position: 2.25`. The `LegUpperR 0.5 > 0.45`
warning is different and is OUR bug: `STANDING_POSE["LegUpperR"]=0.5` exceeds the
real max `0.45`, and the symmetric hardcoded limit `(-1.57,1.57)` fails to clamp
it. Fixes B (pose) and A (runtime limits) both address it. The mock
(`tests/conftest.py`) never emits these warnings — it is confirmed NOT the
source.

Note on signs for STANDING_POSE (fix B): because the model mirrors left/right
with sign-flipped ranges, a right-side crouch value must be NEGATIVE where its
range is `[-1.77, 0.45]` (e.g. `LegUpperR ≈ -0.6`) and its left mirror POSITIVE
within `[-0.5, 1.68]` (e.g. `LegUpperL ≈ +0.6`); knees flex toward the one
reachable side (R toward +, L toward −). This matches the user's observation
that the robot is only stable when flexed/kneeling.

---

# Ordered implementation items

- [ ] 1. Correct the `JOINT_LIMITS` fallback table in config.py to the real
      Darwin-OP PROTO ranges from the finding above, and annotate it. Replace the
      symmetric placeholders with the per-joint `(min, max)` tuples in the table
      (legs/ankles/pelvis/neck/head exact; arms: `ArmUpperL (-2.25,0.77)`,
      `ArmLowerL (-1.18,1.63)` and sign-flipped mirrors for R; shoulders keep a
      conservative `(-3.14,3.14)` fallback). Add a comment stating these are a
      FALLBACK only — authoritative limits are read at runtime (item 3) and
      printed by the diagnostic mode (item 7) — and that the PROTO is remote so
      values cannot be reparsed offline.
      Files: controllers/darwin_common/config.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q tests/test_darwin_robot.py tests/test_config.py` — all pass except nothing new should fail. NOTE: `test_clamp_joint_keeps_margin` and `test_gait_targets_stay_within_safe_limits` read `JOINT_LIMITS["LegLowerR"]`; they must be updated in item 8 in the SAME change so the suite stays green — run the full item-8 verify after item 8.

- [ ] 2. Fix `config.STANDING_POSE` to a balanced crouch within the real limits,
      mirrored and margin-safe. Set leg angles to a modest flexion keeping the
      torso near z≈0.23–0.25: suggested starting values (document each as
      "tune in Webots"): `LegUpperR ≈ -0.6` (within [-1.77,0.45]),
      `LegUpperL ≈ +0.6` (within [-0.5,1.68]); `LegLowerR ≈ +1.2`
      (within [-0.03,2.25]), `LegLowerL ≈ -1.2` (within [-2.25,0.03]);
      `AnkleR ≈ -0.6` (within [-1.22,1.39]), `AnkleL ≈ +0.6` (within [-1.39,1.22]);
      `PelvR/PelvL/PelvYR/PelvYL = 0.0`; keep arms separated but inside the new
      limits (`ArmUpperR ≈ -0.5`→ check sign vs mirror, `ArmUpperL ≈ +0.5`,
      `ArmLowerR ≈ -0.6`, `ArmLowerL ≈ +0.6`, shoulders `±0.8`), `Neck/Head/Foot=0`.
      Every value must sit at least `JOINT_LIMIT_MARGIN` inside its real range so
      clamping never triggers. Add a comment block listing these as Webots-tunable
      starting points.
      Files: controllers/darwin_common/config.py
      Verify: covered by the full suite in item 8; standalone sanity
      `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -c "import sys; sys.path.insert(0,'controllers/darwin_common'); import config; print(config.STANDING_POSE)"` runs without error.

- [ ] 3. Add runtime motor-limit reading in `_LocomotionBackend`
      (darwin_robot.py). In `_fetch_motors`, after resolving each motor, read its
      real bounds with `getMinPosition()`/`getMaxPosition()` guarded by
      `hasattr`+`try/except`, and store them in a new `self.motor_limits: Dict[str,
      Tuple[float,float]]` (fall back to `config.JOINT_LIMITS.get(name)`, else treat
      as unbounded). Initialize `self.motor_limits = {}` in `__init__` before
      `_fetch_motors()`.
      Files: controllers/darwin_common/darwin_robot.py
      Verify: covered by item 8 full suite; `test_locomotion_noop_when_no_motors_resolve`
      must still pass (no motors ⇒ empty `motor_limits`, no exception).

- [ ] 4. Refactor clamping to use backend-provided limits while keeping the
      module-level config fallback. Add a backend method
      `clamp(name, angle)` that clamps against `self.motor_limits[name]` (using
      `config.JOINT_LIMIT_MARGIN`), falling back to the module-level
      `_clamp_joint` when the name is absent. Keep the existing module-level
      `_clamp_joint(name, angle)` (used by `_GaitEngine` and by tests) unchanged in
      signature and behavior against `config.JOINT_LIMITS`. Have
      `_LocomotionBackend.apply_targets` route each angle through
      `self.clamp(name, angle)` as the final guard before `setPosition`, so real
      runtime limits win in Webots while pure-config paths (the engine, tests)
      still work. Preserve the "never command exactly the limit" rule
      (`[min+margin, max-margin]`); if margin exceeds the range, command the range
      center.
      Files: controllers/darwin_common/darwin_robot.py
      Verify: covered by item 8 full suite; specifically
      `test_gait_targets_stay_within_safe_limits` and
      `test_clamp_joint_keeps_margin` (post item-8 update) pass.

- [ ] 5. Make `_GaitEngine` leg-sign logic consistent with the mirrored pose.
      Because `STANDING_POSE` now uses sign-mirrored L/R values, review the
      sinusoidal additions in `compute_targets` so the oscillation stays INSIDE
      each real one-sided range after adding to the new base pose (e.g. knee
      flexion must push R toward + and L toward −, never past `0.03`/`-0.03`). Keep
      the 180° leg phase offset and lateral sway. Do not change the public API
      (`set_amplitudes`, `advance`, `reset_phase`, `compute_targets`,
      `standing_pose`) so existing engine tests keep their contract. The final
      clamp (item 4 / `_clamp_joint`) is the safety net, but amplitudes should be
      chosen so clamping is rarely hit during normal gait.
      Files: controllers/darwin_common/darwin_robot.py (and if needed the
      `GAIT_*_AMPLITUDE` constants in config.py, documented as tunable)
      Verify: covered by item 8; `test_left_and_right_legs_are_180_out_of_phase`,
      `test_stand_sets_nominal_pose`, `test_gait_targets_stay_within_safe_limits`
      all pass.

- [ ] 6. Recalibrate fall detection (config.py + darwin_robot.py). In config.py:
      set `FALL_HEIGHT_Z = 0.14` (clearly below the standing z≈0.23–0.25, above the
      collapsed z≈0.075), keep `ACCEL_CENTER = 512.0`, keep
      `ACCEL_FALL_TOLERANCE` but ADD `FALL_ACCEL_PERSIST_STEPS = 100` (mirrors the
      official Walk.cpp persistence), all commented as Webots-tunable. In
      `DarwinRobot`: add an instance counter (e.g. `self._accel_out_of_band = 0`)
      initialized in `__init__`; in `is_fallen()` the accelerometer branch must
      INCREMENT the counter when the deviation exceeds tolerance and RESET it to 0
      otherwise, returning fall-from-accel ONLY after
      `>= config.FALL_ACCEL_PERSIST_STEPS` consecutive out-of-band ticks — a single
      noisy reading must not flip `fallen`. Keep the height branch as an immediate
      primary signal and keep the yaw-ignoring tilt branch (pure z-axis rotation is
      not a fall). Net required behavior: standing/crouched z≈0.23–0.25 with a
      single noisy accel reading ⇒ `is_fallen()==False`; collapsed z≈0.075 ⇒ `True`.
      Files: controllers/darwin_common/config.py,
      controllers/darwin_common/darwin_robot.py
      Verify: covered by item 8. Existing `test_is_fallen_threshold`,
      `test_is_fallen_true_on_low_height`, `test_is_fallen_true_on_strong_tilt`
      must still pass; item 8 adds a persistence-specific test.

- [ ] 7. Make `STABILIZATION_STEPS` meaningfully long and configurable, and
      document it. Raise `config.STABILIZATION_STEPS` from `20` to a value that lets
      the pose settle (e.g. `80`), with a comment that it is Webots-tunable and that
      a higher value gives the physics more time to settle before the episode/
      command begins. `DarwinRobot.stabilize()` already loops
      `range(int(config.STABILIZATION_STEPS))`; confirm it reads the constant at
      call time (it does) so overrides take effect. Keep it cheap under the mock.
      Files: controllers/darwin_common/config.py
      Verify: covered by item 8; `test_reset_pose_sets_translation` still passes
      (reset_pose runs stabilize()).

- [ ] 8. Update tests in one coherent change: add `FakeMotor` limit getters and
      the new assertions, and realign limit-dependent tests to the corrected
      table. In `tests/conftest.py`: give `FakeMotor` a `getMinPosition()` /
      `getMaxPosition()` pair returning the bounds from `config.JOINT_LIMITS[name]`
      if present, else a wide default (e.g. `(-3.14, 3.14)`); import config at the
      top of conftest if not already. In `tests/test_darwin_robot.py`: update
      `test_clamp_joint_keeps_margin` and `test_gait_targets_stay_within_safe_limits`
      to the new asymmetric `JOINT_LIMITS` (they currently assume `LegLowerR ==
      (-2.25, 2.25)`); add a test that the backend reads runtime limits into
      `motor_limits` and that `clamp` respects them; add a test that a single
      out-of-band accel reading does NOT set `is_fallen()` but
      `FALL_ACCEL_PERSIST_STEPS` consecutive out-of-band readings do. Keep all
      previously passing tests passing.
      Files: tests/conftest.py, tests/test_darwin_robot.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q`
      — expect `N passed, 1 failed` where the ONLY failure is the pre-existing
      `test_wbt.py::test_robot_geometry_and_flags_unchanged`.

- [ ] 9. Upgrade `locomotion_test.py`: command source priority, diagnostic
      motor-limit dump, and resolved standing-z print. Read the command with
      priority `argv[1]` (Webots passes `controllerArgs` as argv) → else
      `os.environ["LOCO_CMD"]` → else `"stand"`, validated against
      `{stand,forward,left,right}`. On startup, after constructing `DarwinRobot`,
      print one line per motor in `config.MOTOR_NAMES` with its REAL
      `getMinPosition()`/`getMaxPosition()` (guarded by `hasattr`+`try/except`;
      fall back to `config.JOINT_LIMITS` or "n/a" for mock motors lacking the
      getters) so the user can capture the true limit table. After
      `reset_pose(...)` stabilization, print the resolved standing z (from
      `get_position()`). Keep the module import-safe without stable-baselines3
      (it already only imports `config` and `darwin_robot`).
      Files: controllers/locomotion_test/locomotion_test.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m py_compile controllers/locomotion_test/locomotion_test.py` succeeds; the full suite from item 8 still passes.

- [ ] 10. Update README/doc notes for the changed knobs and findings.
      Document: the corrected asymmetric joint limits and that real limits are read
      at runtime and dumped by `locomotion_test` diagnostics; the new crouch
      `STANDING_POSE` values as Webots starting points; the fall-detection change
      (`FALL_HEIGHT_Z=0.14`, `FALL_ACCEL_PERSIST_STEPS=100`); larger
      `STABILIZATION_STEPS`; and the `controllerArgs`-first command source for
      `locomotion_test` (how to run `stand|forward|left|right`, e.g. set the robot's
      `controllerArgs "forward"` in the `.wbt`, or `LOCO_CMD=forward`). Correct the
      earlier claim that the knee warnings are purely collapse-against-limit: note
      the `LegUpperR 0.5>0.45` warning was also our mis-set pose/limit, now fixed.
      Files: README.md
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q` unchanged (`1 failed` pre-existing only); README renders as markdown.

- [ ] 11. Final whole-project verification and local commit.
      Run the full suite, the SB3 `check_env` test, and `py_compile` on every
      touched/importing controller, then commit locally (no push, no remote).
      Files: (none new)
      Verify, all must pass (except the one pre-existing `.wbt` failure):
      - `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q`
      - `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q tests/test_race_env.py::test_check_env_sb3`
      - `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m py_compile controllers/darwin_common/config.py controllers/darwin_common/darwin_robot.py controllers/locomotion_test/locomotion_test.py controllers/dqn_agent/dqn_agent.py controllers/ppo_agent/ppo_agent.py controllers/a2c_agent/a2c_agent.py controllers/custom_agent/custom_agent.py`
      Then: `git add -p` the touched files and `git commit` locally on `master`.

---

## What the user must validate and tune inside Webots (no runtime here)

1. Open `pista_atletismo_50m.wbt`, assign `controller "locomotion_test"` to one
   robot, and run `LOCO_CMD=stand` (or `controllerArgs "stand"`). Confirm the
   console prints the 20 motors' real min/max and a resolved standing z near
   0.23–0.25, and that NO `too big/low requested position` warnings appear.
2. Tune `STANDING_POSE` leg angles so the robot stands balanced near z≈0.24 (it
   starts in a modest crouch by design — the user confirmed stability when
   flexed). Adjust `STABILIZATION_STEPS` if it has not settled before motion.
3. Run `forward`, `left`, `right`; tune `GAIT_PERIOD`, `GAIT_X_AMPLITUDE`,
   `GAIT_A_AMPLITUDE`, `GAIT_HIP/KNEE/ANKLE_AMPLITUDE`, `GAIT_LATERAL_SWAY`,
   `MOTOR_VELOCITY` for a stable walk/turn.
4. Confirm `is_fallen()` reports standing=False and a real fall=True; if needed
   tune `FALL_HEIGHT_Z`, `ACCEL_FALL_TOLERANCE`, `FALL_ACCEL_PERSIST_STEPS`,
   `FALL_TILT_THRESHOLD`.
5. Only after a single robot stands/walks/turns reliably, integrate the four
   agents and begin training (per the user's staged plan).
