# Kinematic drag locomotion mode (Plan B) as a config/env-selectable second gait

Commit `b596423` adds a second locomotion mode, `kinematic`, alongside the existing physical sinusoidal gait (`gait`). The project goal is the race and evaluating the RL agents, so a Supervisor-driven "turtle" that slides a standing robot by writing `translation`/`rotation` fields is a deliberate, time-safe alternative to tuning a physical walk. The mode is chosen once per robot via `config.LOCOMOTION_MODE` (default `kinematic`, env-overridable by `LOCOMOTION_MODE`) and resolved through `resolve_locomotion_mode()`. The change is additive: the `_GaitEngine`, the gait constants, and every public method signature stay intact; the kinematic path is guarded behind `mode == "kinematic"` branches. In kinematic mode falling is disabled by design while obstacle/collision penalties and the goal crossing stay active.

Watch for:
- **confirmed** — The commit bundles the concurrent Plan-A gait fix (STANDING_POSE, JOINT_LIMITS, FALL_HEIGHT_Z, FALL_ACCEL_PERSIST_STEPS, STABILIZATION_STEPS, runtime limit reading, yaw-aware tilt) into the same commit as Plan B. The gait code is not rewritten and nothing is clobbered, but Plan A and Plan B are not isolated commits.
- **confirmed** — The kinematic unit tests are committed, but `tests/conftest.py` carries uncommitted mock changes (from the parallel Plan-A work). The Plan-B tests do not depend on them, but the committed tree's conftest is not the tree the suite was verified against.

**Verdict**: APPROVED

## High-level view

The mode is resolved once in `_LocomotionBackend.__init__` and never re-read, so a robot's locomotion is fixed for its lifetime. The action methods (`walk_forward`, `turn_left`, `turn_right`, `stand`) keep their gait lines unchanged and, only in kinematic mode, record a pending `_kin_action` that `update()` consumes.

In kinematic mode each `update()` holds the standing pose on the motors, applies the drag, and resets the node's physics so the robot can neither fall nor accumulate velocity. The drag convention matches the test mock exactly: heading 0 faces +y, forward moves `(-sin θ·d, +cos θ·d, 0)` by `KINEMATIC_LINEAR_STEP`, and a turn adds `±KINEMATIC_ANGULAR_STEP` to the z-rotation. The two tunables equal the mock's `WALK_DELTA`/`TURN_DELTA`, so magnitudes agree under the mock and in real Webots.

The double-displacement problem (mock already moves the robot via `_apply_kinematics`) is handled by skipping the direct field-write whenever the backend exposes `queue_action` — present on the mock, absent on real Webots. Every action still calls `_queue(...)` so the mock continues to drive movement; the field-write is the real-Webots-only path. This split is the subtlest part of the change and is the right call, but it means the real-Webots field-write path has no automated coverage (no Webots in CI).

`is_fallen()` returns `False` early in kinematic mode. `race_env.py` is untouched and stays mode-agnostic: proximity penalties, hard-collision termination, and the `GOAL_Y` win all remain active, confirmed by a new kinematic obstacle-penalty test. `reset_pose()` clears the pending kinematic action so an episode starts without residual drag. The gait path keeps its old behavior; the only textual change in it is a `self.clamp(...)` guard that belongs to the parallel Plan-A limit work, not to this mode.

<details>
<summary>Issues (3)</summary>

1. **Plan A and Plan B share one commit** — The gait-fix edits (pose, joint limits, fall thresholds, runtime limit reading) ride in the same commit as the kinematic mode. Not a defect; note it so the two efforts can be told apart in history if one needs reverting.
2. **conftest mock changes uncommitted** — `tests/conftest.py` has uncommitted `getMinPosition`/`getMaxPosition` additions from the parallel work. The Plan-B tests pass without them, but a clean checkout of this commit alone ships a conftest that differs from the verified one. Confirm the parallel Plan-A commit lands these.
3. **Real-Webots field-write path is unverified** — The `_apply_kinematic_motion` branch that writes `translation`/`rotation` only runs when `queue_action` is absent, i.e. never under the mock, so no test exercises it. Validate once in Webots with `LOCOMOTION_MODE=kinematic` via `locomotion_test` before relying on the drag in the race.

</details>

<details>
<summary>Details</summary>

### Mode routing and the pending-action handshake

`_LocomotionBackend.__init__` resolves `self.mode` once and stores it; the gait engine is still constructed unconditionally. The four action methods keep their original two lines (`engine.set_amplitudes(...)` + `_queue(...)`) and append a single guarded assignment to `self._kin_action` only when `mode == "kinematic"`. `update()` then branches: kinematic holds the pose, applies the recorded action, resets physics, and clears `_kin_action`; the gait branch is the unchanged `engine.advance()` + `apply_targets(engine.compute_targets())`. Clearing the pending action each tick keeps a stale action from repeating into a later tick. The gait branch's only textual change from before is that `apply_targets` now clamps through `self.clamp`, a Plan-A runtime-limits change rather than part of this mode.

### Drag convention and the double-displacement guard

```
heading 0  ->  +y
walk       ->  (x - sin θ·d, y + cos θ·d, z)      d = KINEMATIC_LINEAR_STEP
turn_left  ->  θ += KINEMATIC_ANGULAR_STEP        (z-axis rotation)
turn_right ->  θ -= KINEMATIC_ANGULAR_STEP
θ          =  rot[3] if rot[2] >= 0 else -rot[3]
```

This mirrors the mock's `_apply_kinematics` in `tests/conftest.py`, and the two tunables are set equal to the mock's `WALK_DELTA`/`TURN_DELTA`. Because the mock already moves the robot in `step()`, writing the fields directly would move it twice; `_apply_kinematic_motion` returns early when `hasattr(self.supervisor, "queue_action")`, which is true only under the mock. Real Webots has no `queue_action`, so there the field-write is the sole mover. The forward step reads the current `z` and writes it back unchanged, so only x/y advance and the standing height is preserved; combined with `resetPhysics` each tick the node neither drifts nor gains velocity. The consequence of the guard is that the production field-write path is invisible to the mock-based suite, so its correctness rests on reading the code and a manual Webots check rather than a test.

### Falling disabled, collisions and goal preserved

`is_fallen()` returns `False` as its first statement in kinematic mode. The new `test_is_fallen_false_on_strong_tilt_in_kinematic_mode` and `test_is_fallen_false_on_low_height_in_kinematic_mode` confirm that neither a forced tilt past `FALL_TILT_THRESHOLD` nor a sub-`FALL_HEIGHT_Z` height trips a fall, and `test_gait_mode_still_detects_fall` confirms the gait path still does. `race_env.py` is unchanged, so the proximity penalty, hard-collision termination, and the `GOAL_Y` win are untouched by mode; `test_obstacle_intrusion_penalizes_in_kinematic_mode` asserts `margin_violations >= 1` and `reward < 0` with the kinematic backend active, and `test_check_env_both_modes` runs the SB3 env contract check (not skipped — stable-baselines3 is installed) against both modes.

### Test seams and the uncommitted conftest

The gait/fall tests in `test_darwin_robot.py` and `test_race_env.py` now pin `LOCOMOTION_MODE=gait` before constructing the robot, with an autouse fixture that restores the variable afterward, because the behavior they assert is Plan A that kinematic deliberately disables. The assertions themselves are unchanged. The new kinematic tests pin `LOCOMOTION_MODE=kinematic` the same way. Spot-checking `tests/test_kinematic_locomotion.py` against the committed tree gives 9 passed. The uncommitted `conftest.py` additions (`getMinPosition`/`getMaxPosition` on `FakeMotor`, a `config` import) belong to the parallel Plan-A limit-reading feature; the Plan-B tests fall back to `config.JOINT_LIMITS` when those getters are absent, so they neither require nor break on the committed vs. working-tree conftest. The risk is only that this commit in isolation ships a conftest that is not the one the full suite was verified against.

### Verification evidence

`verification.md` and the commit message record pytest at 82 passed / 1 failed in both the default (kinematic) and `LOCOMOTION_MODE=gait` runs, the single failure being the pre-existing, out-of-scope `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged` (the user moved robots in the `.wbt`). check_env passes in both modes via the parametrized test, and `py_compile` of the four changed sources is clean. The evidence is specific and consistent with the diff; I did not re-run the suites. The one narrow spot-check I ran (the new kinematic test file) passed, confirming the mock-path drag and fall-disable behavior.

### Scope

The commit leaves `race_env.py`, the agent algorithm controllers, `worlds/pista_atletismo_50m.wbt`, and `requirements.md` untouched (verified against the commit's file list). No git remote is configured and nothing was pushed; the work is local-only as required. `locomotion_test.py` honors `LOCOMOTION_MODE` by printing the resolved mode and a kinematic note, and adds a motor-limit diagnostic dump — the behavioral routing stays in `DarwinRobot`, so the test controller gains no mode-specific logic.

</details>

<details>
<summary>File map</summary>

- `controllers/darwin_common/config.py` — adds `LOCOMOTION_MODE`, `KINEMATIC_LINEAR_STEP`, `KINEMATIC_ANGULAR_STEP`, `resolve_locomotion_mode()`, and the Plan A/B comment block (also carries Plan-A pose/limit/fall-threshold edits).
- `controllers/darwin_common/darwin_robot.py` — mode resolved in backend; kinematic `update()` path, `_apply_kinematic_motion`, `_reset_self_physics`, `_kin_action` handshake, `is_fallen()` early return, `reset_pose` clears pending action (also carries Plan-A runtime limit reading and `clamp`).
- `controllers/locomotion_test/locomotion_test.py` — prints resolved mode and kinematic note; adds motor-limit diagnostic dump and controllerArgs command priority.
- `tests/test_kinematic_locomotion.py` — new: drag convention, heading, fall-disabled, reset-clears-action.
- `tests/test_race_env.py` — adds `test_check_env_both_modes`, kinematic obstacle-penalty test, mode-pinning fixture.
- `tests/test_darwin_robot.py` — gait/fall tests pin `LOCOMOTION_MODE=gait`; assertions unchanged.
- `README.md` — documents Plan A vs B, mode selection, the two tunables, by-design drag, disabled falling with active collisions/goal, drag-testing.

Full diff: `git show b596423`.

</details>
