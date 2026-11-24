# Kinematic (Plan B) locomotion refinement for the DARwIn-OP race

The commit reworks the default kinematic drag mode so the Supervisor-dragged robot moves at a human-realistic speed, stops at barriers and other robots instead of passing through them, and slides while visually upright instead of collapsed. Speed moved from a hardcoded 0.25 m/tick (~15.6 m/s) to `config.KINEMATIC_LINEAR_STEP = 0.016` (~1.0 m/s at a 16 ms tick), with the mock and the real Webots path now reading the same constant so velocity is tunable in one place. Collision blocking is decided inside `_LocomotionBackend` by projecting the would-be next (x,y) and cancelling forward motion if it lands within `KINEMATIC_COLLISION_RADIUS` of a barrier or another robot; turning is never blocked. A z-pin to `KINEMATIC_STAND_Z = 0.24` plus yaw-only rotation keeps the torso upright as a best-effort visual approximation. The gait path (Plan A) is untouched and stays opt-in.

Watch for: the upright z-pin is a visual approximation that can only be confirmed in Webots — the author documented this and listed what to check (confirmed). The z-pin is written only on `walk_forward`; turning and idle ticks rely on `resetPhysics()` + `hold_standing()` and do not re-assert z, a minor behavioral asymmetry rather than a defect (confirmed). No Webots runtime exists here, so the four runtime behaviors (speed, stop-at-barrier, upright torso, `is_fallen()` False) remain unverified by execution and rest on the mock tests plus the recorded `check_env` runs (confirmed).

**Verdict**: APPROVED

## High-level view

Speed is now a single tunable. `KINEMATIC_LINEAR_STEP` (0.016) and `KINEMATIC_ANGULAR_STEP` (0.04) carry a documented m/s and deg/s derivation in `config.py`, and both the real displacement in `darwin_robot.py` and the mock's `WALK_DELTA`/`TURN_DELTA` read them, so nothing hardcodes 0.25 anymore. The unit tests assert against the constant, not a literal.

Collision blocking lives in `DarwinRobot`, not the mock. `walk_forward` in kinematic mode calls `_kinematic_blocked_forward()`, which projects the next (x,y) with the same `(-sin θ·d, +cos θ·d)` formula the drag uses and checks it against barriers (`valla*`) and other robots within `KINEMATIC_COLLISION_RADIUS = 0.3`. When blocked, the action is neither queued nor applied, so the real-vs-mock guard is preserved and movement is never double-counted. Turning is deliberately never blocked, so the agent can steer away. The radius sits below `SAFETY_RADIUS` (0.6) and above `HARD_COLLISION_DISTANCE` (0.15), so the proximity penalty still fires before the block and hard-collision termination is unaffected; `race_env` scoring is unchanged.

Visual upright pinning writes `z = KINEMATIC_STAND_Z = 0.24` on each forward drag, keeps rotation as pure yaw about z, commands the standing pose to the motors, and calls `resetPhysics()` every tick. This is explicitly a visual approximation; the config and README spell out what the user must confirm in Webots.

The gait path is byte-for-byte intact, `LOCOMOTION_MODE` default stays `"kinematic"`, public method signatures are unchanged, `is_fallen()` is still False in kinematic, and `.wbt` / `requirements.md` were not touched. The two pre-existing constants kept their values' slots (only values changed) and two new constants were appended.

<details>
<summary>Issues (2)</summary>

1. **Z-pin on forward only** — `_apply_kinematic_motion` writes `KINEMATIC_STAND_Z` only for `walk_forward`; turn and idle ticks don't re-assert z and lean on `resetPhysics()`/`hold_standing()`. Minor asymmetry; confirm in Webots the torso stays at ~0.24 while turning or stationary.
2. **Runtime behaviors unverified** — speed, stop-at-barrier, upright torso and `is_fallen()`=False are validated only by mock tests and `check_env`; no Webots runtime here. Author documented the four manual checks; the user must run them.

</details>

<details>
<summary>Details</summary>

### Speed as a single source of truth

`KINEMATIC_LINEAR_STEP` dropped from 0.25 to 0.016 and `KINEMATIC_ANGULAR_STEP` from 0.20 to 0.04, each with a derivation comment tying the value to the 16 ms tick (0.016 m/tick ÷ 0.016 s = 1.0 m/s; 0.04 rad/tick ÷ 0.016 s ≈ 2.5 rad/s ≈ 143 °/s) and noting the old values were ~15.6 m/s and ~716 °/s (confirmed). The real drag in `_apply_kinematic_motion` reads `config.KINEMATIC_LINEAR_STEP`, and the mock in `conftest.py` sets `WALK_DELTA = config.KINEMATIC_LINEAR_STEP` / `TURN_DELTA = config.KINEMATIC_ANGULAR_STEP`, keeping the mock coherent with the real path and preserving the `WALK_DELTA`/`TURN_DELTA` names for tests that reference them (confirmed). `tests/test_race.py` and `tests/test_race_env.py` now derive goal distances and step counts from the constant rather than assuming 0.25, so the tests won't break when the speed is retuned (confirmed).

### Collision blocking decided in DarwinRobot

```
walk_forward() [kinematic]
  └─ _kinematic_blocked_forward()
        ├─ _self_xy_theta()      → (x, y, θ) of own node
        ├─ project next: nx = x - sinθ·d, ny = y + cosθ·d
        └─ _obstacles_xy()       → barriers (valla*) + other robots (excl. self by identity)
             └─ any hypot((nx,ny),(bx,by)) < KINEMATIC_COLLISION_RADIUS → blocked
  blocked  → _kin_action = None, not queued  (no move, no double-apply)
  clear    → _queue(...) + _kin_action = "walk_forward"
```

The projection reuses the exact drag formula and heading convention (`θ = rot[3] if rot[2] >= 0 else -rot[3]`), so the predicted next position matches what the drag would actually write (confirmed). Self-exclusion is by node identity, not by name, which avoids a robot blocking itself even if names collide (confirmed). Barriers and other robots are gathered through the same module-level scene-scan helpers (`_iter_world_nodes`/`_node_name`/`_node_xy`) that `find_barriers`/`find_others` now use, so the blocking source matches the penalty source. Turning never consults the blocker, so a cornered agent can still rotate to escape (confirmed).

The refactor that lifted the three scene helpers from `DarwinRobot` methods to module level preserves behavior: `find_others` still filters by `name == self.name` and membership in `START_POSITIONS`, and `find_barriers` still matches the `valla` prefix; the helpers themselves stayed defensive against missing root/children (confirmed).

The radius ordering is the load-bearing design point. `HARD_COLLISION_DISTANCE` (0.15) < `KINEMATIC_COLLISION_RADIUS` (0.3) < `SAFETY_RADIUS` (0.6) means the proximity penalty in `race_env` triggers before the drag is blocked, and the hard-collision termination threshold is still reachable, so the block layers on top of the existing scoring without displacing it (confirmed).

### Visual upright pinning and physics reset

On each kinematic tick, `update()` commands the standing pose (`hold_standing`), applies the drag, then calls `resetPhysics()` on the own node so teleport-per-step doesn't accumulate fall velocity or drift (confirmed). The forward drag writes `z = KINEMATIC_STAND_Z` instead of inheriting the possibly-collapsed z, and rotation is written as `[0, 0, 1, θ]`, i.e. pure yaw with zero roll/pitch. The author is explicit, in both `config.py` and the README, that this is a visual approximation and lists the four things to confirm in Webots (speed ~1 m/s, stop-at-barrier with turning allowed, torso at ~0.24, `is_fallen()` False).

The z-pin is asserted only in the `walk_forward` branch. A turn writes rotation but not translation, and an idle tick (action None) writes nothing, so between forward steps the torso height is held only by `resetPhysics()` and the commanded standing pose rather than an explicit z write. This is a minor asymmetry worth confirming in Webots rather than a correctness bug, since the stated goal is a standing slide and the real-Webots pin path is exercised by a dedicated test.

### Test coverage

The new `tests/test_kinematic_locomotion.py` covers the five required behaviors: advance equals `config.KINEMATIC_LINEAR_STEP` (not a literal), a barrier directly ahead cancels the move with position unchanged, turning still changes heading in that same blocked configuration, another robot ahead also blocks, and the real-Webots path (a supervisor wrapper that hides `queue_action`) pins z to `KINEMATIC_STAND_Z` from a collapsed start and leaves rotation as pure yaw (confirmed by reading the test file). The real-Webots seam is a test-only construct, but here it is the only way to exercise the direct `setSFVec3f` path that the mock otherwise short-circuits, so it earns its place. Pre-existing kinematic tests (mode resolution, heading-following advance, `is_fallen()` False on tilt and low height, gait-still-detects-fall counter-check, reset clears pending action) remain.

Not verified by any test: the four Webots runtime behaviors themselves. These cannot be exercised without the simulator and are deferred to the documented manual checklist.

### Scope

`.wbt` and `requirements.md` are absent from the changed-file list (confirmed). The gait branch of `update()` (`engine.advance()` + `apply_targets`) and the gait lines in `walk_forward`/`turn_*` (`set_amplitudes(...)`) are unchanged, `LOCOMOTION_MODE` default remains `"kinematic"`, and the two existing constants kept their identity with only their values changed while two new constants were appended rather than renumbering anything (confirmed). `tests/test_race.py` relocates its test robot to a free lane (x=5.0) because barriers now block forward motion — a necessary consequence of the new blocking, explained in the test comment, not an unrelated change (confirmed).

</details>

<details>
<summary>File map</summary>

- `controllers/darwin_common/config.py` — speed constants retuned with derivations; `KINEMATIC_COLLISION_RADIUS` and `KINEMATIC_STAND_Z` added; Webots-check notes.
- `controllers/darwin_common/darwin_robot.py` — scene-scan helpers lifted to module level; collision block (`_kinematic_blocked_forward`/`_obstacles_xy`/`_self_xy_theta`) added; `walk_forward` cancels blocked forward; drag pins z and yaw.
- `tests/conftest.py` — mock reads config speed constants; `WALK_DELTA`/`TURN_DELTA` derive from them.
- `tests/test_kinematic_locomotion.py` — five new refined-mode tests (a–e).
- `tests/test_race.py` / `tests/test_race_env.py` — goals/steps derived from the constant; test robot moved to a free lane.
- `README.md` — realistic speed, obstacle blocking, upright pin, gait opt-in unchanged, single-place tunables.
- `.agents/tasks/kinematic-refine/verification.md` — recorded verification evidence.

Full diff: `git show 1cf727b`.

</details>
