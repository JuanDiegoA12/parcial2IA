# Ordered-checkpoint lap reward for the oval DARwIn-OP race

The race track is an oval whose start coincides with its finish, so the previous "progress toward y=9" reward was all but guaranteed without actually lapping the course. This change replaces the fixed-goal reward with an ordered checkpoint lap: the robot must reach cp1..cp7 in order and only then return to cp0 to win; crossing y=9 without completing the sequence no longer counts. Checkpoints are read from the world by name (`checkpoint1..checkpoint7`), the observation grows from 5 to 7 floats to expose direction/distance to the next checkpoint and lap progress, and an off-track penalty keeps the robot near the ideal racing line. The same `CheckpointTracker` state machine drives both training reward and the race winner detector, so they agree by construction.

**Watch for:** observation dims 4 and 5 (`lap_progress`, `next_cp_index_norm`) carry identical values — `target_ordinal_norm()` just calls `lap_progress()` (confirmed); it is a wasted dimension, not a bug. The `.wbt` is modified in the working tree but is NOT part of this commit (confirmed), so it is out of scope for this review. Retraining is mandatory since the observation shape changed (acknowledged in README and commit).

**Verdict**: APPROVED

## High-level view

The lap logic lives in one place. `CheckpointTracker` is a pure-Python state machine that, given the ordered target list, tracks which checkpoint is next, measures distance reduction toward it, fires a discrete bonus on arrival within `CHECKPOINT_RADIUS`, and marks the lap done when the final target (cp0) is reached after the rest. Both `DarwinRaceEnv` and `RaceWinnerDetector` instantiate it from the same ordered coordinate list, which is what makes reward and victory agree.

Checkpoint coordinates come from the world at runtime. `DarwinRobot.find_checkpoints()` scans scene nodes named `checkpoint<N>`; `ordered_checkpoints()` synthesizes cp0 as `(0, GOAL_Y)`, orders by `CHECKPOINT_ORDER`, and degrades to the documented `config.CHECKPOINT_COORDS` fallback with a `RuntimeWarning` when a node is missing. The config map is a fallback for mock/tests and graceful degradation, not the source of truth.

The win condition is now lap completion, not line crossing. `success` in the env is `tracker.done`; the detector records finish order as robots complete the lap. Crossing y=9 early produces no win because cp0 is only credited as the last target after cp1..7.

Off-track is penalized continuously by the point-to-segment distance from the current target's predecessor to the target, beyond `OFF_TRACK_TOLERANCE`, scaled by `OFF_TRACK_PENALTY_WEIGHT`, without terminating the episode. The existing safe-learning penalties (proximity, hard-collision termination, fall, per-step) are untouched.

Only the next checkpoint is ever credited. Distance, reached-test, and bonus all key off `current_target()`, so reaching a later checkpoint early earns nothing until the sequence advances to it.

<details>
<summary>Issues (2)</summary>

1. **Duplicate observation dimension** — `target_ordinal_norm()` returns exactly `lap_progress()`, so obs[3] and obs[4] are always equal; consider making obs[4] the normalized target index (`reached_count / (len-1)` or the raw cp id) or dropping it. Non-blocking.
2. **Off-track segment lags by one tick on arrival** — on the tick a checkpoint is reached, the credited `off_segment` is the segment just completed rather than the new one, because it is captured before `reached_count` advances. Cosmetic; one-tick effect, no correctness impact.

</details>

<details>
<summary>Details</summary>

## CheckpointTracker as the shared source of truth

The core design move is extracting lap bookkeeping into `CheckpointTracker`, a state machine with no Webots or numpy dependency. It holds the ordered target list, a `reached_count`, a `prev_dist`, and a `done` flag. `update(px, py)` returns the per-tick progress (reduction in distance to the current target), whether the current target was reached, whether the lap closed, the running count, and the current `(predecessor, target)` segment for the off-track penalty.

```
ordered_checkpoints()  ->  [cp1, cp2, ..., cp7, cp0]
                             |                    |
          DarwinRaceEnv -----+                    +----- RaceWinnerDetector
          (reward, success)        same list        (finish order, winner)
```

Both consumers build the tracker from the same ordered list, so a robot that earns the win-closing reward in training is exactly a robot the detector would rank as finishing. This removes the prior risk of reward and race logic drifting apart. `run_race_until_winner` sources the list from the first robot's `ordered_checkpoints()` (same runtime scan as the env) with a config fallback, then hands it to the detector, which keeps one tracker per robot name.

When `getter()` raises, `checkpoints` falls back to `None` and the detector rebuilds from config; the loop only ever consults the first robot before breaking. That is the documented intent and is acceptable for a homogeneous robot set.

## Name-discovered checkpoints with synthesized cp0

`find_checkpoints()` scans the world for nodes whose name is `checkpoint` followed by digits and returns `{index: (x, y)}`, mirroring the existing `find_barriers()`/`find_others()` pattern. cp0 is deliberately not a named node; `ordered_checkpoints()` synthesizes it at `(0, GOAL_Y)`. Missing checkpoints fall back to `config.CHECKPOINT_COORDS` and raise a `RuntimeWarning` naming the missing indices, so a mislabeled or absent node degrades gracefully rather than crashing. The config map documents itself as a fallback and matches the ball translations the user placed in the world.

## Win only on completing the ordered lap

`success = tracker.done`, and `done` is set only when `reached_count` reaches the end of the target list, whose final element is cp0. Because cp0 is credited last, a robot sitting near y=9 at the start cannot win: cp1 is its first target, far away. The detector mirrors this — it registers a finisher only when that robot's tracker reports `done`. The test suite exercises both the positive case (full lap wins and terminates) and the negative case (crossing the goal plane without the sequence does not win), for both the env and the detector.

## Off-track penalty on the racing line

The penalty is the point-to-segment distance from the robot to the line between the current target's predecessor and the target, using a standard clamped projection with a degenerate-segment guard. Deviation beyond `OFF_TRACK_TOLERANCE` is scaled by `OFF_TRACK_PENALTY_WEIGHT` and subtracted from the step reward without terminating — matching the user's request to discourage leaving the track rather than ending the episode. On the tick a checkpoint is reached, the credited segment is the one just completed (captured before the count advances); this is a one-tick cosmetic lag, not a correctness issue.

## Observation growth 5 -> 7

The observation now carries `[dist_next_cp, angle_next_cp, heading, lap_progress, next_cp_index_norm, dist_obstacle, dist_other]`, with `observation_space` bounds extended to 7 (`low` gains two `0.0` entries, `high` stays all ones). Distance and angle now reference the current target rather than the fixed goal plane, and the angle convention (`atan2(-dx, dy)`) is consistent with the heading convention. After the lap completes, `_target_point()` falls back to the last target so the observation stays finite and in range.

The fifth and sixth entries are the concern: `target_ordinal_norm()` returns `lap_progress()` verbatim, so the two carry identical values on every tick. The model receives one redundant input. Harmless for correctness but worth tightening — obs[4] could instead expose the raw or normalized target index so the two signals differ.

## Config, metrics, and plotting

`config.py` adds the tunables (`CHECKPOINT_RADIUS`, `CHECKPOINT_PROGRESS_WEIGHT`, `CHECKPOINT_BONUS`, `OFF_TRACK_PENALTY_WEIGHT`, `OFF_TRACK_TOLERANCE`, `CHECKPOINT_ORDER`, `TOTAL_TARGETS`, `CHECKPOINT_COORDS`), each commented, and extends `RaceParams` with matching fields using `default_factory` for the mutable order/coords so defaults stay backward-compatible. `agent_runner.run_race` now reports `checkpoints_reached` (max reached in the episode) alongside the preserved keys, and `plot_results.summarize` adds the aggregation only when the column is present, so old CSVs still load. `Discrete(3)` actions, both locomotion modes, and `is_fallen` are untouched.

## Test coverage

New `tests/test_checkpoints.py` covers the tracker, the segment-distance helper, and integration; `test_darwin_robot.py` adds `find_checkpoints`/`ordered_checkpoints`; `test_race_env.py` moves to the 7-dim observation, progress toward the next checkpoint, full-lap win, and the no-win-on-bare-crossing case, plus `check_env` in both locomotion modes; `test_race.py` is rewritten around "first to complete the lap" and adds the crossing-without-lap negative; `test_config.py` asserts the new constants and `RaceParams` fields; `conftest.py` seeds `checkpoint1..7` in the mock world. The verification record reports 109 passed with the single tolerated pre-existing failure (`tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`), caused by the user's robot repositioning in the `.wbt` and explicitly out of scope.

Not tested at this level: behavior when a checkpoint node is genuinely absent from the world (the `RuntimeWarning` fallback path) is covered by the fallback map but not asserted to emit the warning in a dedicated test — minor, since the degradation path is simple.

</details>

<details>
<summary>File map</summary>

- `controllers/darwin_common/config.py` — new checkpoint tunables and `RaceParams` fields (backward-compatible defaults).
- `controllers/darwin_common/darwin_robot.py` — `find_checkpoints()` and `ordered_checkpoints()` (name scan, cp0 synthesis, graceful fallback).
- `controllers/darwin_common/race_env.py` — `CheckpointTracker`, 7-dim observation, checkpoint progress/bonus and off-track reward, lap-completion success.
- `controllers/darwin_common/race.py` — winner = first to complete the lap, reusing `CheckpointTracker` per robot; entry points preserved.
- `controllers/darwin_common/agent_runner.py` — `checkpoints_reached` metric.
- `experiments/plot_results.py` / `run_experiments.py` — optional `checkpoints_reached` column.
- `tests/conftest.py` — mock gains `checkpoint1..7`.
- `tests/test_checkpoints.py` (new) + updates to `test_darwin_robot.py`, `test_race_env.py`, `test_race.py`, `test_config.py`, `test_agent_runner.py`, `test_custom_agent.py`, `test_experiments.py`.
- `README.md` — oval/same-start-finish problem, ordered lap, name-discovered checkpoints, 7-dim observation, tunables, retraining notice.

Full diff: `git show 6240f3c`.

</details>
