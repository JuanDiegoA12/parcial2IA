# Implementation Plan — Checkpoint-based reward on the oval track

Goal: replace the "progress toward fixed finish line y=9" scheme with an ORDERED-CHECKPOINT
lap (cp0 → cp1 → … → cp7 → cp0, one lap). Observation grows 5 → 7 dims, reward becomes
checkpoint-based with an off-track penalty, and winner detection requires completing the
ordered lap. Shared lap-state machine used by both `race_env.py` and `race.py`.

## Design decisions (confirmed with user; recorded here, not re-decided)

- **Checkpoints discovered by name at runtime** (like `valla*`): `DarwinRobot.find_checkpoints()`
  scans the scene for nodes named `checkpoint<N>` (N = 1..7) and returns `{index: (x, y)}`;
  index 0 is synthesized from `GOAL_Y` as `(0.0, GOAL_Y)`. Known coordinates kept in config
  only as a documented fallback for the mock/tests, never as the source of truth.
- **Ordered required sequence** (one lap): `[1, 2, 3, 4, 5, 6, 7, 0]` — the robot starts at
  cp0, so there are **8 targets** to reach in order (cp1..cp7, then return to cp0). `TOTAL_TARGETS = 8`.
- **Reward**: continuous `progress_weight * (prev_dist_to_target − dist_to_target)` toward the
  CURRENT target only + a discrete `checkpoint_bonus` when the current target is reached (within
  `CHECKPOINT_RADIUS`), then advance the target. WIN (terminate + `goal_bonus`) only when the
  final target (return to cp0) is reached after all of 1..7. Crossing y=9 without completing the
  ordered checkpoints must NOT win. Keep all existing safe-learning penalties unchanged.
- **Off-track penalty** (user decision 8=b): penalize distance from the LINE SEGMENT between the
  current target's predecessor and the current target, beyond `OFF_TRACK_TOLERANCE`, scaled by
  `OFF_TRACK_PENALTY_WEIGHT`. Deterministic, simple.
- **7-dim observation** (chosen, see item 2): `[dist_to_next_cp, angle_to_next_cp, heading,
  lap_progress, next_cp_index_norm, dist_nearest_obstacle, dist_nearest_other_robot]`.
- **One lap only.** Keep `Discrete(3)` actions, both locomotion modes, and `is_fallen()` as-is.

## Chosen 7-dim observation vector and normalized bounds

| idx | component                     | raw → normalized                                  | low  | high |
|-----|-------------------------------|---------------------------------------------------|------|------|
| 0   | distance to next checkpoint   | `min(dist / max_dist, 1.0)`                       | 0.0  | 1.0  |
| 1   | angle to next checkpoint      | `clip(angle / π, -1, 1)`                          | -1.0 | 1.0  |
| 2   | heading                       | `clip(heading / π, -1, 1)`                        | -1.0 | 1.0  |
| 3   | lap_progress                  | `reached_count / TOTAL_TARGETS` ∈ [0,1]           | 0.0  | 1.0  |
| 4   | next_cp_index_norm            | `target_ordinal / TOTAL_TARGETS` ∈ [0,1]          | 0.0  | 1.0  |
| 5   | distance to nearest obstacle  | `1.0` if none else `min(dist / max_dist, 1.0)`    | 0.0  | 1.0  |
| 6   | distance to nearest other bot | `1.0` if none else `min(dist / max_dist, 1.0)`    | 0.0  | 1.0  |

`target_ordinal` = how many targets have been reached so far (0..TOTAL_TARGETS); it is the
position of the current target in the ordered sequence, so index 4 and index 3 are equal in this
design but serve as two distinct "how far along the lap" signals as requested (progress vs. the
normalized index of the next target). `low = [0, -1, -1, 0, 0, 0, 0]`, `high = ones(7)`.

## Checkpoint lap-state machine (shared by env and race.py)

Introduce ONE small helper class in `race_env.py` (importable by `race.py`) so both agree:

`CheckpointTracker` (plain Python, no Webots, no numpy dependency beyond math):
- Built from an ordered list of checkpoint coordinates `targets = [(x1,y1)…(x7,y7),(0,GOAL_Y)]`
  plus `radius` (CHECKPOINT_RADIUS). Also stores `predecessors`: for target k, the previous
  point in the full ordered path (cp0 for the first target, else the prior checkpoint), used by
  the off-track segment distance.
- State: `reached_count` (int, 0..len(targets)), `prev_dist` (float|None), `done` (bool).
- `reset()` → `reached_count=0, prev_dist=None, done=False`.
- `current_target()` → coords of the current target, or None when `done`.
- `current_segment()` → `(predecessor_point, target_point)` for the off-track penalty.
- `update(px, py)` → returns a dict `{progress, reached (bool), done (bool),
  reached_count, target_ordinal, off_segment}`; advances `reached_count` when within `radius`,
  sets `done=True` when the last target (return to cp0) is reached. `progress` is the reduction
  in distance to the current target since the previous call (0 on the first call / right after an
  advance). On the tick a target is reached, also credit progress toward it.
- `lap_progress()` → `reached_count / len(targets)`; `target_ordinal_norm()` →
  `reached_count / len(targets)` (current target's normalized index).

Both `DarwinRaceEnv` and `RaceWinnerDetector` own a `CheckpointTracker` built from the SAME
`DarwinRobot.find_checkpoints()` result (ordered via `config.CHECKPOINT_ORDER`), so the win
condition and the reward agree by construction.

## Off-track segment-distance penalty formula

Point `P=(px,py)`; segment from `A` (predecessor) to `B` (current target). Let `AB = B−A`,
`t = clamp(dot(P−A, AB) / dot(AB, AB), 0, 1)`, projection `Q = A + t·AB`,
`d = hypot(px−Qx, py−Qy)`. Penalty applied only when `d > OFF_TRACK_TOLERANCE`:
`reward -= OFF_TRACK_PENALTY_WEIGHT * (d − OFF_TRACK_TOLERANCE)`. If `AB` is degenerate
(zero length) use point distance to `A`. Deterministic; no termination.

---

- [ ] 1. Add checkpoint/lap tunables and `RaceParams` fields to config.py.
      Add, with Spanish comments: `CHECKPOINT_RADIUS: float = 1.0` (m, reach threshold / win
      radius around cp0), `OFF_TRACK_PENALTY_WEIGHT: float = 2.0`, `OFF_TRACK_TOLERANCE: float = 1.0`
      (m deviation allowed from the racing line before penalty — documented as ~lane half-width,
      tunable), `CHECKPOINT_PROGRESS_WEIGHT: float = 10.0` (reuse semantics of PROGRESS_WEIGHT for
      the toward-next-checkpoint term), `CHECKPOINT_BONUS: float = 20.0` (discrete per-checkpoint
      reward), `CHECKPOINT_ORDER: tuple = (1, 2, 3, 4, 5, 6, 7, 0)` (ordered required lap),
      `TOTAL_TARGETS = len(CHECKPOINT_ORDER)` (= 8), and a documented fallback map
      `CHECKPOINT_COORDS: Dict[int, Tuple[float, float]]` with cp1..cp7 at the real .wbt coords
      (cp1 (8.79,7.98), cp2 (13.51,1.49), cp3 (8.79,-7.96), cp4 (1.46,-8.97), cp5 (-8.5,-8.25),
      cp6 (-13.48,1.49), cp7 (-8.5,8.21)) plus cp0 = (0.0, GOAL_Y) — marked "fallback para
      mock/tests; la fuente autoritativa es find_checkpoints en runtime". Extend the `RaceParams`
      dataclass with backward-compatible defaulted fields mirroring these:
      `checkpoint_radius`, `off_track_penalty_weight`, `off_track_tolerance`,
      `checkpoint_progress_weight`, `checkpoint_bonus`, `checkpoint_order`, `checkpoint_coords`
      (the last two via `default_factory`). Leave GOAL_Y, START_POSITIONS, all kinematic/gait
      constants, and the existing reward weights untouched.
      Files: controllers/darwin_common/config.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_config.py` — passes (existing config
      tests unchanged); `./.venv/bin/python -c "import sys; sys.path.insert(0,'controllers/darwin_common'); import config; p=config.RaceParams(); print(p.checkpoint_radius, p.checkpoint_order, config.TOTAL_TARGETS)"`
      prints `1.0 (1, 2, 3, 4, 5, 6, 7, 0) 8`.

- [ ] 2. Add `find_checkpoints()` to DarwinRobot.
      Add a method mirroring `find_barriers()`: scan `_iter_world_nodes(self.backend)`, match
      nodes whose name matches `checkpoint<N>` (N an integer 1..), parse the integer index, and
      return `{index: (x, y)}` using `_node_xy`. Add a companion
      `ordered_checkpoints(order=config.CHECKPOINT_ORDER, goal_y=...)` helper (or build the ordered
      list in the env — see item 3) that: synthesizes index 0 as `(0.0, goal_y)`, pulls 1..7 from
      the scan, falls back to `config.CHECKPOINT_COORDS` for any missing index, logs a
      `warnings.warn(..., RuntimeWarning)` if any of 1..7 were missing, and returns the ordered
      coordinate list following `CHECKPOINT_ORDER`. Do NOT hardcode coords as source of truth.
      Files: controllers/darwin_common/darwin_robot.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_darwin_robot.py` — passes including the
      new checkpoint test added in item 7.

- [ ] 3. Add `CheckpointTracker` and rewrite the reward/obs in race_env.py.
      In `controllers/darwin_common/race_env.py`: (a) add the `CheckpointTracker` class described
      in "Checkpoint lap-state machine" above (pure Python, importable by race.py); (b) in
      `DarwinRaceEnv.__init__` build the ordered checkpoint list from
      `self.robot.find_checkpoints()` + synthesized cp0 (reuse the item-2 helper), construct a
      `CheckpointTracker(targets, radius=self.params.checkpoint_radius)` stored on the env, and set
      `observation_space` to the 7-dim Box from the table above (`low=[0,-1,-1,0,0,0,0]`,
      `high=np.ones(7)`); (c) `reset()` resets the tracker and `_prev_dist` bookkeeping (now owned
      by the tracker); (d) rewrite `step()`: compute `px,py`, call `tracker.update(px,py)`,
      `reward = params.checkpoint_progress_weight * result["progress"] − params.step_penalty`,
      add `params.checkpoint_bonus` when `result["reached"]`, add the off-track penalty from
      `tracker.current_segment()` using the formula above, keep the obstacle/other-robot proximity,
      hard-collision, and fall penalties/terminations exactly as today, and set
      `terminated = result["done"]` with `reward += params.goal_bonus` on done. `success` in `info`
      becomes `result["done"]` (full lap), NOT `py >= goal_y`; also add `info["checkpoints_reached"]
      = result["reached_count"]`; (e) rewrite `_get_obs()` to emit the 7-dim vector (replace
      `_dist_to_goal`/`_angle_to_goal` with distance/angle to `tracker.current_target()`; when
      `done`, target is None → use the last target coords so obs stays finite and in-bounds). Keep
      `_nearest_distance` as-is.
      Files: controllers/darwin_common/race_env.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_race_env.py` — passes after item 6 test
      updates; `./.venv/bin/python -c "import sys;[sys.path.insert(0,p) for p in ('controllers/darwin_common',)]; import os; os.sys.path" ` not needed — covered by the SB3 check_env test (item 6).

- [ ] 4. Rewrite winner detection in race.py to use the ordered lap.
      Replace the "first across y=GOAL_Y" logic: `RaceWinnerDetector` keeps one `CheckpointTracker`
      per robot (keyed by name), built lazily from the first positions seen or from an injected
      ordered-checkpoint list. `update(positions)` feeds each robot's `(x,y)` to its tracker and
      records a robot in `finish_order` only when its tracker reports `done` (completed the ordered
      lap); winner = first to complete. Keep the public signatures: `RaceWinnerDetector(goal_y=…)`
      still accepts `goal_y` (used to synthesize cp0) and gains an optional
      `checkpoints`/`order`/`radius` parameter with config defaults; `run_race_until_winner(robots,
      max_steps, goal_y=…, action_fn=None)` keeps its signature and now builds the ordered
      checkpoint list from the first robot's `find_checkpoints()` (fallback to config coords),
      passing it to the detector so env and race agree. Update the module docstring to describe the
      lap-based win.
      Files: controllers/darwin_common/race.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_race.py` — passes after item 6 test updates.

- [ ] 5. Add checkpoint nodes to the conftest mock world.
      In `tests/conftest.py`, add a `_CHECKPOINTS` dict (`"checkpoint1"`..`"checkpoint7"` → real
      .wbt translations with z=0.0625) and register them as `FakeNode`s in `FakeWebots.__init__`'s
      shared `_world` (alongside `_ROBOT_STARTS` and `_BARRIERS`), so `find_checkpoints()` finds
      them via the scene scan. Do not change `_apply_kinematics`, `WALK_DELTA`/`TURN_DELTA`, or the
      robot/barrier entries. Checkpoint names must not start with `valla` (they don't), so
      `find_barriers()` still returns exactly 12.
      Files: tests/conftest.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_darwin_robot.py::test_find_barriers_returns_twelve`
      — still passes (12 barriers); the new checkpoint test (item 7) passes.

- [ ] 6. Update existing tests that assert the OLD 5-dim obs or OLD y=9 win.
      These tests encode the replaced behavior and MUST change:
      - `tests/test_race_env.py`: `test_spaces_shape` and `test_reset_returns_obs_and_info` assert
        `shape == (5,)` → update to `(7,)`. `test_forward_progress_is_positive` assumes progress
        toward y=9 — reposition / re-aim so moving reduces distance to the first target (cp1 at
        (8.79,7.98)) or assert `info["progress"]` via a short deterministic move; keep the intent
        (a forward move yields positive progress/reward). `test_crossing_goal_grants_bonus_and_
        terminates` currently wins by `py>=goal_y` — rewrite to a "complete lap" test: drive the
        robot (or directly advance the env's `CheckpointTracker` through all targets, e.g. by
        teleporting `getSelf()` to each ordered checkpoint and stepping) and assert
        `info["success"]`/`terminated` only after the final return to cp0, and assert a plain
        crossing of y=9 without the ordered checkpoints does NOT set success. `test_obstacle_
        intrusion_penalizes*` and `test_truncation_at_max_steps` stay valid but must stop relying on
        `env._prev_dist`/`_dist_to_goal` — use `tracker` state or just assert the penalty/truncation
        outcome.
      - `tests/test_race.py`: `test_detector_first_to_cross_wins` and
        `test_detector_same_tick_most_advanced_wins` assert first-across-y=9 → rewrite to feed each
        robot a full ordered-checkpoint sequence and assert the winner is the first to COMPLETE the
        lap; keep `test_detector_uses_config_goal_by_default` (cp0 derives from GOAL_Y).
        `test_run_race_until_winner_on_mock` relied on `agente propio` at y=11.99 winning instantly
        (past y=9) — rewrite so the winner is whoever completes the lap (drive one robot through the
        checkpoints, or shrink the problem deterministically). `test_run_race_until_winner_progresses`
        similarly updated to complete the lap. Keep `test_config.py` as-is (GOAL_Y unchanged).
      Files: tests/test_race_env.py, tests/test_race.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_race_env.py tests/test_race.py` — all pass,
      including `test_check_env_sb3` and `test_check_env_both_modes` with the new 7-dim obs.

- [ ] 7. Add new unit tests for checkpoints, tracker, reward, and off-track penalty.
      Add:
      - `tests/test_darwin_robot.py`: `test_find_checkpoints_returns_seven` — `find_checkpoints()`
        returns indices 1..7 at the real coords; and a test that the ordered-checkpoint helper
        synthesizes cp0 at `(0, GOAL_Y)` and orders by `CHECKPOINT_ORDER`.
      - New `tests/test_checkpoints.py` (CheckpointTracker unit tests, no Webots): advancing through
        targets in order increases `reached_count`; reaching a LATER checkpoint early is NOT
        credited (only the current target counts); `done` becomes True only after the final return
        to cp0; `progress` is the per-step distance reduction toward the current target; off-track
        segment distance is 0 on the line, > tolerance off the line. Also assert the env applies
        `OFF_TRACK_PENALTY_WEIGHT` when the robot is driven off the current segment beyond tolerance
        (reward strictly lower than on-line).
      - `tests/test_config.py` (append): assert the new constants exist with expected defaults and
        `TOTAL_TARGETS == 8`, and `RaceParams` carries the new fields (override non-destructive).
      Files: tests/test_darwin_robot.py, tests/test_checkpoints.py, tests/test_config.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_checkpoints.py tests/test_darwin_robot.py tests/test_config.py`
      — all pass.

- [ ] 8. Keep experiments + plots working; add checkpoint columns safely.
      `experiments/run_experiments.py`: `run_race` metrics already flow through `agent_runner.run_race`;
      add `checkpoints_reached` by having `agent_runner.run_race` read `info["checkpoints_reached"]`
      (max over the episode) and return it, then append `"checkpoints_reached"` to
      `CSV_COLUMNS` and to the row dict in `run_config`. `plot_results.summarize` must keep working:
      add `checkpoints_reached=("checkpoints_reached","mean")` to the agg (guard with
      `if "checkpoints_reached" in df.columns` so old CSVs still load). Existing columns and the
      two figures stay. Note `success` now means "completed the lap".
      Files: controllers/darwin_common/agent_runner.py, experiments/run_experiments.py,
      experiments/plot_results.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_agent_runner.py tests/test_experiments.py tests/test_plots.py`
      — pass after item 9 updates the schema assertion.

- [ ] 9. Update tests that assert the exact experiments CSV schema and run_race keys.
      `tests/test_experiments.py::test_csv_schema_constant` asserts `CSV_COLUMNS` EXACTLY → add
      `"checkpoints_reached"` to the expected list. `tests/test_agent_runner.py` (`test_run_race_
      returns_metric_keys`, `test_run_race_truncates_without_crossing`) and
      `tests/test_custom_agent.py::test_run_race_returns_metrics` assert `set(metrics) == {success,
      steps, cumulative_reward, falls, collisions}` → add `"checkpoints_reached"` to the expected
      set. These tests also assume `success` from crossing a low `goal_y`; since success now means
      completing the lap, adjust them to assert truncation/metric-keys rather than lap success where
      a full lap is not driven (keep the key-shape assertions; relax the `success is True`
      expectation that depended on y-crossing).
      Files: tests/test_experiments.py, tests/test_agent_runner.py, tests/test_custom_agent.py
      Verify: `./.venv/bin/python -m pytest -q tests/test_experiments.py tests/test_agent_runner.py tests/test_custom_agent.py`
      — all pass.

- [ ] 10. Update README for the checkpoint lap (Spanish prose, English identifiers).
      Revise the intro (lines ~5), the `race_env.py`/`race.py` bullet descriptions (~30-32), and the
      win/reward sections (~96, ~131, ~385-386): the winner is now the first robot to COMPLETE the
      ordered checkpoint lap `cp0 → cp1 → … → cp7 → cp0`, the observation is 7-dim (list the vector),
      the reward is checkpoint-progress + per-checkpoint bonus + off-track penalty, and note that
      agents must be RETRAINED (obs dim changed 5 → 7). Document the new config tunables
      (`CHECKPOINT_RADIUS`, `OFF_TRACK_PENALTY_WEIGHT`, `OFF_TRACK_TOLERANCE`, `CHECKPOINT_ORDER`,
      `CHECKPOINT_COORDS`) in the "Parámetros ajustables" section (~184). Do NOT edit the .wbt or
      requirements.md.
      Files: README.md
      Verify: manual read — `grep -n "checkpoint\|7 dim\|lap\|vuelta" README.md` shows the new
      content; no code command needed (docs only).

- [ ] 11. Full verification pass (no Webots).
      Run the whole suite and the SB3 env check in BOTH locomotion modes, and py_compile the changed
      files.
      Files: (none — verification only)
      Verify:
      1. `./.venv/bin/python -m pytest -q` → all pass EXCEPT the one known pre-existing failure
         `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged` (out of scope; do NOT touch).
      2. `./.venv/bin/python -c "import sys; [sys.path.insert(0,p) for p in ('controllers/darwin_common',)]; from tests.conftest import FakeWebots; from stable_baselines3.common.env_checker import check_env; from race_env import DarwinRaceEnv; fw=FakeWebots(); fw.install(); [check_env(DarwinRaceEnv(fw.make_supervisor(self_name='ppo'), robot_name='ppo'), skip_render_check=True) for _ in [0]]"`
         for kinematic (default); repeat with `LOCOMOTION_MODE=gait ./.venv/bin/python …` — both pass
         (this is also covered by `tests/test_race_env.py::test_check_env_both_modes`).
      3. `./.venv/bin/python -m py_compile controllers/darwin_common/config.py controllers/darwin_common/darwin_robot.py controllers/darwin_common/race_env.py controllers/darwin_common/race.py controllers/darwin_common/agent_runner.py experiments/run_experiments.py experiments/plot_results.py tests/conftest.py`
         → exits 0.
      Then commit locally only (`git add` the changed files + this plan + new test file; `git commit`).
      NEVER push/fetch/pull.

## Tests that assert OLD behavior and MUST be updated (summary)

- `tests/test_race_env.py`: `test_spaces_shape`, `test_reset_returns_obs_and_info` (5-dim →
  7-dim); `test_crossing_goal_grants_bonus_and_terminates` (y=9 win → lap-complete win);
  `test_forward_progress_is_positive`, `test_obstacle_intrusion_penalizes*`,
  `test_truncation_at_max_steps` (stop relying on `_dist_to_goal`/`_prev_dist`).
- `tests/test_race.py`: `test_detector_first_to_cross_wins`, `test_detector_same_tick_most_
  advanced_wins`, `test_run_race_until_winner_on_mock`, `test_run_race_until_winner_progresses`
  (first-across-y=9 → first-to-complete-lap).
- `tests/test_experiments.py::test_csv_schema_constant` (add `checkpoints_reached`).
- `tests/test_agent_runner.py` + `tests/test_custom_agent.py` run_race metric-key sets (add
  `checkpoints_reached`; relax y-crossing success assumptions).
- NOT touched: `tests/test_config.py` core assertions (GOAL_Y=9 stays), `tests/test_wbt.py`
  (the known failure, out of scope).

## Assumptions / gaps noted

- Checkpoints 1..7 are assumed present in the world (confirmed in the .wbt). If any are missing at
  runtime, `find_checkpoints` degrades to the config fallback and logs a warning.
- `index 3` and `index 4` of the obs are numerically equal by this design (both encode "fraction of
  the lap done / index of the next target"); kept as two signals per the task's suggested vector.
  If a richer second signal is wanted later, index 4 could become normalized Euclidean lap distance
  — out of scope now.
- The mock moves the robot kinematically; lap-completion tests drive it by teleporting `getSelf()`
  to successive checkpoints (the mock already exposes `set_translation`), which is deterministic and
  Webots-free.
