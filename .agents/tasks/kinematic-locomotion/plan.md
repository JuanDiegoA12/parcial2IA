# Implementation Plan — Kinematic "drag/slide" locomotion (Plan B)

Add a second, kinematic locomotion mode selectable by a config flag, as a time-safe
alternative to the physical sinusoidal gait (Plan A). The point of the project is the
RACE and evaluating RL agents, so a kinematic differential-drive "turtle" driven by the
Supervisor is an accepted design choice. All work is **additive**: a concurrent workflow
is editing the SAME files to tune Plan A, so do not rewrite or reformat existing gait code
or change gait constants.

## Context discovered during exploration (read before implementing)

- `darwin_common/config.py` and `darwin_common/darwin_robot.py` are BOTH being edited by
  the concurrent Plan-A workflow and are currently dirty (unstaged `git diff`). The code on
  disk already differs from earlier states: `config.py` now has mirrored asymmetric
  `JOINT_LIMITS`, a reworked `STANDING_POSE`, `FALL_HEIGHT_Z=0.14`, `STABILIZATION_STEPS=80`,
  and a new `FALL_ACCEL_PERSIST_STEPS`; `darwin_robot.py` now has `_clamp_to_limits`,
  `_LocomotionBackend.motor_limits`, `clamp`, and `_read_motor_limits`. **Re-read both files
  at implementation time** and build on whatever state they are in — do not assume the
  snippets in this plan are current. If a hunk no longer applies cleanly, prefer the minimal
  additive edit and record what was touched.
- Heading convention (do not change): `get_orientation()` returns `angle if rot[2]>=0 else
  -angle` around z; heading 0 faces +y. Forward displacement = `(-sin(theta)*d, +cos(theta)*d, 0)`.
  Turning adds `+/-` to the z-axis rotation angle. This mirrors the mock's `_apply_kinematics`
  in `tests/conftest.py` (`WALK_DELTA=0.25`, `TURN_DELTA=0.20`, `theta = rot[3] if rot[2]>=0 else -rot[3]`).
- The mock drives movement under tests via `queue_action(...)` + `step()`→`_apply_kinematics()`.
  Real Webots has NO `queue_action` and NO physics teleport helper; there the Supervisor must
  write `translation`/`rotation` fields directly.
- Existing high-level API that MUST stay intact: `DarwinRobot.walk_forward/turn_left/turn_right/
  stand/tick/stabilize/reset_pose/get_position/get_orientation/is_fallen`, and
  `_LocomotionBackend.walk_forward/turn_left/turn_right/stand/update/hold_standing/apply_targets`.
- Baseline test state: `.venv/bin/python -m pytest -q` → **58 passed, 1 failed**. The only
  failure is `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged` (user moved robots in
  the `.wbt`; out of scope, leave it). Treat "58 passed, 1 known failure" as green.

## Key design decisions (recorded per instructions)

- **D1 — avoid double displacement under the mock.** In kinematic mode the action methods set
  internal intended-motion state AND still call `_queue(...)` (so existing mock movement tests
  keep passing). The field-write that produces the drag happens **only when the backend does NOT
  expose `queue_action`** (i.e. real Webots); under the mock the queue+`_apply_kinematics` already
  produces the motion with the same sign/axis convention. This prevents the displacement being
  applied twice under the mock. Rationale: the task requires both the direct field-write (real
  Webots) and the `_queue` call (mock), and the two must agree — gating the field-write on
  "no queue_action" keeps them from stacking while leaving each environment with exactly one mover.
- **D2 — make config steps equal to the mock deltas.** Set `KINEMATIC_LINEAR_STEP = 0.25` (=
  `WALK_DELTA`) and `KINEMATIC_ANGULAR_STEP = 0.20` (= `TURN_DELTA`). Because of D1 the mock moves
  the robot by exactly these magnitudes, so the new unit tests can assert the robot moved by
  `KINEMATIC_LINEAR_STEP` / changed heading by `KINEMATIC_ANGULAR_STEP` and pass under the mock
  while the same constants drive the real-Webots field-write. Rationale: keeps mock and real
  behavior numerically consistent and makes the new tests deterministic without touching the mock.
- **D3 — mode plumbing lives in `_LocomotionBackend`, resolved once at construction.** Read the
  mode from `config.LOCOMOTION_MODE`, overridable by env var `LOCOMOTION_MODE`, via a small helper
  `config.resolve_locomotion_mode()`. `DarwinRobot.is_fallen()` consults the backend's resolved
  mode. Rationale: locomotion already lives in the backend; centralizing the flag there keeps
  `DarwinRaceEnv` and the agents untouched and keeps one source of truth for the mode.
- **D4 — hold standing + reset physics each kinematic tick, defensively.** In kinematic mode each
  tick commands the standing pose to the motors (skipped cleanly if no motors, e.g. the mock) and
  calls `resetPhysics()` on the self node if available (absent under the mock → skipped). Only the
  teleport via Supervisor fields moves the robot. Rationale: matches decision 3 (robot looks
  upright and does not accumulate a fall) without breaking the mock.
- **D5 — this work does not decompose into separable features.** The change is one coherent
  additive edit threaded through four tightly-coupled files plus tests and docs, and it must be
  serialized against a concurrent writer on the same files. Splitting it across parallel FEAT
  coders would race the git index and the same files. So this is a single plan; the existing
  implement/review loop runs it. The workflow tail is NOT restructured.

## Additive strategy against the concurrent writer

- Only ADD new constants/sections/branches. Never delete, renumber, or reformat existing
  constants or gait code. Append new config constants in a clearly marked new block at the END of
  the module-level constants (before `@dataclass RaceParams`), not interleaved with gait constants.
- In `darwin_robot.py`, add a new branch inside existing methods guarded by `self.mode ==
  'kinematic'`; the `== 'gait'` path must execute the EXACT current code unchanged.
- Keep every existing public signature. Add no required constructor args.
- If a listed line/anchor has shifted due to concurrent edits, find the nearest equivalent anchor
  and make the smallest additive change; note it in the implementation summary.

---

# Implementation Plan

- [ ] 1. Add Plan-A/Plan-B comment block and the three kinematic constants to config, plus a mode
      resolver, without touching existing constants.
      Append a new section after `STABILIZATION_STEPS` / before `--- Normalización ---`:
      `LOCOMOTION_MODE: str = "kinematic"` (default), `KINEMATIC_LINEAR_STEP: float = 0.25` (meters
      per walk_forward tick; equals the mock's `WALK_DELTA`), `KINEMATIC_ANGULAR_STEP: float = 0.20`
      (radians per turn tick; equals the mock's `TURN_DELTA`), and a function
      `resolve_locomotion_mode() -> str` that returns `os.environ.get("LOCOMOTION_MODE",
      LOCOMOTION_MODE)` lowercased and validated to `{"gait","kinematic"}` (fallback to
      `LOCOMOTION_MODE` if unrecognized). Add `import os`. Include a comment block explaining Plan A
      (sinusoidal motor gait) vs Plan B (kinematic drag via Supervisor) and that Plan B is a
      deliberate race/agent-evaluation mode. Do NOT remove or renumber existing constants.
      Files: controllers/darwin_common/config.py
      Verify: `.venv/bin/python -m pytest -q tests/test_config.py` passes; `.venv/bin/python -c
      "import sys; sys.path.insert(0,'controllers/darwin_common'); import config;
      print(config.LOCOMOTION_MODE, config.KINEMATIC_LINEAR_STEP, config.KINEMATIC_ANGULAR_STEP,
      config.resolve_locomotion_mode())"` prints `kinematic 0.25 0.2 kinematic`.

- [ ] 2. Add kinematic mode to `_LocomotionBackend` and `DarwinRobot` so the gait path stays
      byte-for-byte unchanged and the kinematic path drags the standing robot.
      In `_LocomotionBackend.__init__`, after `self.engine = _GaitEngine()`, add
      `self.mode = config.resolve_locomotion_mode()`. Add a private helper
      `_apply_kinematic_motion(self, action)` that, ONLY when the backend does NOT expose
      `queue_action` (real Webots; see decision D1), reads the self node's `translation`/`rotation`
      fields via `self.supervisor.getSelf()` and writes the new pose: for `walk_forward` add
      `(-sin(theta)*KINEMATIC_LINEAR_STEP, +cos(theta)*KINEMATIC_LINEAR_STEP, 0)` to translation
      keeping z; for `turn_left`/`turn_right` set rotation `[0,0,1, theta +/- KINEMATIC_ANGULAR_STEP]`;
      using `theta = rot[3] if rot[2]>=0 else -rot[3]`. In `walk_forward/turn_left/turn_right/stand`,
      keep the existing `set_amplitudes(...)` + `_queue(...)` lines unchanged and, when
      `self.mode == "kinematic"`, record the pending kinematic action on `self._kin_action`.
      Add a kinematic branch to `update()`: when `self.mode == "kinematic"`, call `hold_standing()`
      (motors to standing pose; no-op if no motors), `_apply_kinematic_motion(self._kin_action)`,
      reset physics on the self node if `resetPhysics` exists, then clear `self._kin_action`; when
      `self.mode == "gait"`, run the EXISTING `engine.advance()` + `apply_targets(...)` unchanged.
      In `DarwinRobot.is_fallen()`, add an early `if self.locomotion.mode == "kinematic": return
      False` guard at the top. Do not change any other method body.
      Files: controllers/darwin_common/darwin_robot.py
      Depends on: item 1 (uses the new config symbols).
      Verify: `.venv/bin/python -m pytest -q tests/test_darwin_robot.py` passes (existing gait and
      mock-movement tests still green); `.venv/bin/python -m py_compile
      controllers/darwin_common/darwin_robot.py` exits 0.

- [ ] 3. Confirm `race_env.py` needs no behavioral change and is safe in both modes.
      Read `reset()`/`step()`: proximity/collision/goal logic is mode-agnostic and must stay exactly
      as today. `reset_pose()` → `stabilize()` already drives `hold_standing()`, which is a no-op
      without motors and under kinematic mode, so no edit is expected. Make a change ONLY if a test
      from item 6 reveals an actual error; if so, the change must be additive and must not alter
      proximity/collision/goal behavior. Record "no change needed" in the implementation summary if
      nothing is required.
      Files: controllers/darwin_common/race_env.py (likely unchanged)
      Depends on: item 2.
      Verify: `.venv/bin/python -m pytest -q tests/test_race_env.py` passes in the default
      (kinematic) mode; `LOCOMOTION_MODE=gait .venv/bin/python -m pytest -q tests/test_race_env.py`
      also passes.

- [ ] 4. Make `locomotion_test.py` honor the locomotion mode and print mode + position/heading so a
      single robot can be drag-tested.
      Read the resolved mode via `config.resolve_locomotion_mode()` and print it in the startup log
      lines (alongside robot name and command). No behavioral branching is required in this
      controller because `DarwinRobot` already routes by mode; just surface the mode so the user can
      confirm Plan B is active. Keep the existing `LOCO_CMD`/`LOCO_ROBOT` handling and the per-step
      position/heading/accel/fallen log unchanged.
      Files: controllers/locomotion_test/locomotion_test.py
      Depends on: item 1.
      Verify: `.venv/bin/python -m py_compile controllers/locomotion_test/locomotion_test.py` exits 0;
      `.venv/bin/python -m pytest -q` shows no new failures (this controller is import-safe; do NOT
      launch Webots).

- [ ] 5. Verify the SB3 env contract in BOTH modes.
      No code change — this item exists to gate on `check_env`. Add nothing unless a failure appears;
      if one does, fix it with a minimal additive change in `darwin_robot.py`/`race_env.py` only.
      Files: none (verification gate)
      Depends on: items 2-3.
      Verify: `.venv/bin/python -c "import sys; sys.path.insert(0,'controllers/darwin_common'); import
      tests.conftest"` is NOT the path here — instead run the dedicated check via item 6's test
      `test_check_env_both_modes`. Direct manual check: run `tests/test_race_env.py::test_check_env_sb3`
      under both `LOCOMOTION_MODE=kinematic` and `LOCOMOTION_MODE=gait` and confirm both pass (skips
      cleanly if stable-baselines3 is absent).

- [ ] 6. Add unit tests for kinematic mode, reusing the mock's heading/displacement convention.
      Create `tests/test_kinematic_locomotion.py` using the `fake_webots` fixture. The tests set the
      mode via monkeypatching the env var `LOCOMOTION_MODE` (set it BEFORE constructing the robot so
      `_LocomotionBackend.__init__` resolves it) — use `monkeypatch.setenv("LOCOMOTION_MODE",
      "kinematic")`. Cover: (a) `walk_forward`+`tick` with heading 0 increases y by
      `config.KINEMATIC_LINEAR_STEP` (approx) and leaves x unchanged; (b) `turn_left`+`tick` increases
      heading by `KINEMATIC_ANGULAR_STEP` and `turn_right`+`tick` decreases it by the same;
      (c) `is_fallen()` returns False in kinematic mode even after `supervisor.force_tilt(
      FALL_TILT_THRESHOLD+0.5)` and even after forcing z below `FALL_HEIGHT_Z`; (d) a gait-mode
      sanity test (`monkeypatch.setenv("LOCOMOTION_MODE","gait")`) confirming `is_fallen()` still
      returns True under a strong tilt (mode routing works both ways). Also add, in
      `tests/test_race_env.py`, a parametrized `test_check_env_both_modes` or extend existing tests to
      assert obstacle/other-robot proximity penalties still fire in kinematic mode (reuse the setup in
      `test_obstacle_intrusion_penalizes`, with kinematic mode active). If stable-baselines3 is
      present, assert `check_env` passes in kinematic mode.
      Files: tests/test_kinematic_locomotion.py (new), tests/test_race_env.py (additive)
      Depends on: items 1-2.
      Verify: `.venv/bin/python -m pytest -q tests/test_kinematic_locomotion.py` passes;
      `.venv/bin/python -m pytest -q tests/test_race_env.py` passes.

- [ ] 7. Update `README.md` (Spanish prose, English identifiers) documenting Plan A vs Plan B.
      Add a subsection under "Locomoción" explaining: Plan A = sinusoidal motor gait (existing),
      Plan B = kinematic drag that slides the standing robot via the Supervisor BY DESIGN for the
      race / agent-evaluation goal; how to select the mode (`config.LOCOMOTION_MODE` or env
      `LOCOMOTION_MODE`, values `gait`/`kinematic`, default `kinematic`); the two tunables
      (`KINEMATIC_LINEAR_STEP`, `KINEMATIC_ANGULAR_STEP`); that in kinematic mode falling is disabled
      (`is_fallen()` returns False) but collision/proximity penalties and crossing `GOAL_Y` still
      count; and how to drag-test one isolated robot with `locomotion_test` (e.g.
      `LOCOMOTION_MODE=kinematic LOCO_CMD=forward webots pista_atletismo_50m.wbt`). Do not remove the
      existing gait documentation.
      Files: README.md
      Depends on: items 1-6.
      Verify: `grep -n "LOCOMOTION_MODE" README.md` shows the new section (doc-only item; correctness
      is confirmed by reading, not by tests).

- [ ] 8. Full-suite regression and compile gate.
      Run the whole test suite in the default (kinematic) mode and in gait mode, and byte-compile the
      changed sources. Confirm the ONLY failing test is the known out-of-scope
      `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`.
      Files: none (final gate)
      Depends on: items 1-7.
      Verify: `.venv/bin/python -m pytest -q` → all pass except the one known `test_wbt` failure;
      `LOCOMOTION_MODE=gait .venv/bin/python -m pytest -q` → same (gait behavior unchanged);
      `.venv/bin/python -m py_compile controllers/darwin_common/config.py
      controllers/darwin_common/darwin_robot.py controllers/darwin_common/race_env.py
      controllers/locomotion_test/locomotion_test.py` exits 0. Do NOT launch Webots.

## Gaps / assumptions

- The `.wbt` already fails `test_wbt.py::test_robot_geometry_and_flags_unchanged` (user moved
  robots); this is pre-existing and explicitly out of scope — leave it failing.
- Assumed the mock is not modified (its `WALK_DELTA`/`TURN_DELTA` stay 0.25/0.20). Decision D2 ties
  the kinematic config steps to those values so mock-based magnitude assertions hold. If the
  concurrent workflow changes the mock deltas, update `KINEMATIC_LINEAR_STEP`/`KINEMATIC_ANGULAR_STEP`
  to match (and note it), since the real-Webots field-write and the mock must stay numerically
  consistent for the tests.
- Assumed real Webots is the only environment lacking `queue_action`; decision D1 uses that as the
  discriminator for applying the direct field-write. This matches the current mock, which always
  provides `queue_action`.
- Do NOT modify the `.wbt` world or `requirements.md`, and do NOT change any agent algorithm logic.
