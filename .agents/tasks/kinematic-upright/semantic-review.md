# Kinematic (Plan B) upright-stance fix for the Webots DARwIn-OP racer

The kinematic drag mode slid the robot on its back while levitating, because the upright correction (yaw-pure rotation + z pin) was written only inside the `walk_forward` branch, and the motors were being commanded the gait's crouch (`STANDING_POSE`) every tick. This change moves the posture correction into a new `_apply_upright_pose()` that runs every tick for all actions (stand/walk/turn), and commands a new straight-legged `KINEMATIC_STANDING_POSE` to the motors instead of the crouch. Movement (x,y), speed, collision blocking, `is_fallen()=False`, the z pin to 0.24, and the entire gait path (Plan A) are preserved. The design deliberately splits posture (own-node rotation + z, written unconditionally) from movement (x,y/θ, still behind the `queue_action` guard) so the posture fix is observable under the test mock while real-Webots movement is not duplicated.

Watch for: the x,y/θ drag path in real Webots is untestable under the mock by design (confirmed), so the per-action posture tests exercise `_apply_upright_pose` but not the real-Webots walk/turn field writes — those remain covered only by the pre-existing real-path test. One minor scope note: the commit bundled the task-tracking docs (`plan.md`, `verification.md`) alongside the four intended files (confirmed), which the verification note said would be a 4-file commit.

**Verdict**: APPROVED

## High-level view

The posture fix hinges on one design decision: separate the upright write from the movement write. `_apply_kinematic_motion` keeps its `queue_action` early-return (so under the mock the toy kinematics in `FakeSupervisor.step()` produce x,y without duplication), while the new `_apply_upright_pose()` writes rotation and z directly to the own node with no such guard, running every tick for every action. That is what makes stand/turn stay upright, not just walk, and it is what makes the z-pin and yaw-pure rotation observable in tests.

The rotation rewrite uses the exact heading convention of `get_orientation` (`theta = rot[3] if rot[2] >= 0 else -rot[3]`, written back as `[0,0,1,theta]`), so an already-upright node keeps its heading unchanged after the rewrite. `get_orientation` reads `rot[2]`/`rot[3]` and returns the same value, so headings and turn accumulation are unaffected.

The z pin moved out of the `walk_forward` branch, which now writes only x,y and preserves the current z; `_apply_upright_pose` fixes z to `KINEMATIC_STAND_Z` the same tick, so the numeric result is identical but now applies to all actions. `KINEMATIC_STAND_Z` stays at 0.24 with an expanded tuning comment.

Motors now receive `KINEMATIC_STANDING_POSE` (legs near-straight, knees R+0.1/L−0.1, everything else neutral) via the new `hold_kinematic_standing()`, clamped through the existing `clamp`. All 20 angles sit inside the real asymmetric `JOINT_LIMITS` with ≥0.05 margin, so clamping never fires. The gait's `hold_standing()`/`STANDING_POSE` are untouched and still used on the gait path and in stabilize/reset.

`is_fallen()` still returns `False` in kinematic unchanged; `resetPhysics` keeps its mock-vs-real guard. Collision blocking, the movement formula, and the step constants are all untouched.

<details>
<summary>Issues (2)</summary>

1. **Real-path field writes untested under mock** — the x,y/θ writes in `_apply_kinematic_motion` short-circuit under the mock, so the new per-action tests cover `_apply_upright_pose` only; the real-Webots walk/turn writes rely on the one pre-existing real-path test. Acceptable by design, but worth knowing the mock cannot catch a regression in that branch.
2. **Commit scope wider than stated** — the commit includes `plan.md` and `verification.md` in addition to the four intended files; the verification note described a 4-file commit. No behavioral impact; adjust the note or the commit contents if strict scope matters.

</details>

<details>
<summary>Details</summary>

### Posture split: `_apply_upright_pose` vs `_apply_kinematic_motion`

The root fix is architectural. Previously the upright write (z pin + yaw-pure rotation) lived inside the `walk_forward` branch of `_apply_kinematic_motion`, so stand and the two turn actions never re-straightened the node and physics let the torso fall between ticks. The new `_apply_upright_pose()` pulls that write out and runs it every tick in the kinematic branch of `update()`, for every action.

The method reads the own node's rotation, derives `theta = rot[3] if rot[2] >= 0 else -rot[3]`, and writes back `[0.0, 0.0, 1.0, theta]` — stripping any roll/pitch while keeping the yaw. It then reads translation, keeps x,y, and sets `z = config.KINEMATIC_STAND_Z`. It does not move x,y. The ordering in `update()` (posture before movement) is immaterial: posture touches only z/rotation, movement only x,y.

Why this split matters is the mock boundary. `_apply_kinematic_motion` early-returns when the supervisor exposes `queue_action` (true under `FakeSupervisor`), because the toy kinematics in `FakeSupervisor._apply_kinematics` produce the x,y/θ drag. If the posture write stayed inside that guard, it would be invisible to tests. `_apply_upright_pose` has no `queue_action` guard and writes the node fields directly, which `FakeNode` exposes, so the z pin and yaw-pure rotation are observable under the mock and verified per action. In real Webots both paths coexist without clobbering: movement writes x,y, posture writes z and rotation.

### Heading preserved through the rotation rewrite

`get_orientation` computes `angle if axis_z >= 0 else -angle` from `rot[2]`/`rot[3]`. `_apply_upright_pose` writes `rot[2]=1.0` and `rot[3]=theta` using the same sign convention, so for a node already carrying a `[0,0,1,θ]` rotation (the steady state after any prior tick) the heading is unchanged. The turn branches in `_apply_kinematic_motion` then read theta fresh and add `±KINEMATIC_ANGULAR_STEP`, so turn accumulation is intact. The `test_upright_pose_preserves_heading_on_turn` test confirms `turn_left` still advances the heading by exactly `KINEMATIC_ANGULAR_STEP`.

A note on the per-action test seed: it starts from `[1,0,0,0.5]` (roll about x) and asserts the post-tick rotation is `[0,0,1,·]`. Because `rot[2]=0 >= 0`, theta becomes 0.5 and the roll is reinterpreted as yaw — which is exactly the point, since the pre-existing collapsed/tilted states are what the fix must straighten. In normal operation the incoming rotation is already yaw-pure, so no heading is lost.

### z pin relocation

The `walk_forward` branch changed from writing `[x, y, KINEMATIC_STAND_Z]` to `[x, y, z]` (current z preserved), because `_apply_upright_pose` now pins z the same tick. The numeric outcome is identical (z ends at `KINEMATIC_STAND_Z`) but now for stand and turns too, not just walk. Two mock-path assertions in the walk tests were updated from `z1 == z0` to `z1 == KINEMATIC_STAND_Z`, correctly reflecting that z is now pinned under the mock where it previously was not.

### Standing pose: upright stance vs gait crouch

`update()` now calls `hold_kinematic_standing()` instead of `hold_standing()` in the kinematic branch. The new helper commands `config.KINEMATIC_STANDING_POSE` through `self.clamp`, keeping the gait's `hold_standing`/`STANDING_POSE` entirely for the gait path (and stabilize/reset). `KINEMATIC_STANDING_POSE` sets knees to R+0.1 / L−0.1 (mirrored signs matching the asymmetric `LegLowerR∈[-0.02,2.25]` / `LegLowerL∈[-2.25,0.03]` limits) and everything else to 0.0. Every angle sits at least `JOINT_LIMIT_MARGIN` (0.05) inside its real limit, so the clamp never alters the commanded values — the verification note's `OUT_OF_RANGE []; OK 20` check and the limit table both confirm this. `test_kinematic_commands_standing_pose_not_gait_crouch` asserts the motors receive the clamped `KINEMATIC_STANDING_POSE` and that it differs from the gait crouch for `LegLowerR/LegUpperR/AnkleR/LegLowerL`.

### Unchanged invariants

`is_fallen()` still returns `False` immediately in kinematic mode (unchanged line). The collision path (`_kinematic_blocked_forward`, `KINEMATIC_COLLISION_RADIUS`, barrier/other lookups), the advance formula `(-sinθ·d, +cosθ·d)`, and the step constants are untouched. `_reset_self_physics` keeps its guard that skips `resetPhysics` when the node lacks it (the mock), so it stays a no-op in tests and active in Webots. The gait branch of `update()` (`engine.advance()` + `apply_targets(compute_targets())`) is byte-identical. `README.md` documents the per-tick upright write, `KINEMATIC_STANDING_POSE` in the tunables table, and the `KINEMATIC_STAND_Z` raise/lower tuning guidance, in Spanish prose with English identifiers, with no edits to the Plan A gait section. `requirements.md`, the agents, experiments, and `worlds/pista_atletismo_50m.wbt` carry no changes in this commit (the `.wbt` remains unstaged, out of scope).

### Verification evidence

The coder's note records the full pytest run (96 passed, 1 failed — the single failure being the pre-existing `test_wbt.py::test_robot_geometry_and_flags_unchanged` caused by `.wbt` drift, outside this task's scope and matching the recorded baseline of 90 passed/1 failed), `check_env` passing in both gait and kinematic modes, `py_compile` of the touched files, and the pose limit check. The diff traces cleanly against every judgment point, so no spot-check was warranted.

</details>

<details>
<summary>File map</summary>

- `controllers/darwin_common/config.py` — new `KINEMATIC_STANDING_POSE` (upright stance, 20 motors); expanded `KINEMATIC_STAND_Z` tuning comment (value unchanged at 0.24).
- `controllers/darwin_common/darwin_robot.py` — new `_apply_upright_pose()` (per-tick yaw-pure rotation + z pin); `walk_forward` branch now writes x,y only; `update()` kinematic branch commands the upright pose and calls `_apply_upright_pose`; new `hold_kinematic_standing()`; `hold_standing` docstring clarified.
- `tests/test_kinematic_locomotion.py` — two walk-test z assertions updated to the pinned value; new per-action z-pin/yaw-pure test (parametrized), heading-preservation-on-turn test, and commanded-pose-vs-crouch test.
- `README.md` — documents per-tick upright posture, `KINEMATIC_STANDING_POSE` tunable, and `KINEMATIC_STAND_Z` raise/lower guidance.
- `.agents/tasks/kinematic-upright/{plan.md,verification.md}` — task-tracking docs bundled in the commit.

Full diff: `git show HEAD` (commit `1a1b839`) or `git diff HEAD~1 HEAD`.

</details>
