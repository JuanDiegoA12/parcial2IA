# Implementation Plan — DARwIn-OP Race RL (Webots R2025a)

WORKDIR = `/home/juan/Documentos/parcial2 IA` (no git worktree). All paths below are absolute under WORKDIR unless noted. Comments/docs may be Spanish; identifiers in English.

## Context grounded in the world file

- `pista_atletismo_50m.wbt` has 4 `Darwin-op` robots, all sharing `DEF DARWIN`, each `supervisor TRUE`, `selfCollision TRUE`, `controller "<none>"`:
  - `name "DQN"` @ `0.11 9.95 0.24`
  - `name "agente propio"` @ `0.21 11.99 0.34`
  - `name "ppo"` @ `0.21 7.88 0.34`
  - `name "actor - critic"` @ `0.21 6.04 0.34`
- Because all 4 share DEF `DARWIN`, `getFromDef("DARWIN")` cannot disambiguate. Each controller gets ITSELF via `supervisor.getSelf()`, and finds the OTHER robots and barriers by scanning root world children and matching the `name` field. This is the single most important robot-layer design decision.
- `DEF START_FINISH Pose` @ translation `(0, 9, 0.014)`, a box `0.18 (x) × 8 (y) × 0.008`. The finish line is the plane `y = 9` (robots race along the +y / -y axis, not x). Goal line y is a config constant; "reaching goal" = crossing the configured goal y with the right sign of progress.
- 12 `WorkBarrier` obstacles ("valla 1" .. "valla 1(11)"), x in [-16.7, 16.7], clustered near `y ≈ -0.11` and `y ∈ [-12, -6]`.
- Webots R2025a; DARwIn-OP PROTO is EXTERNPROTO from cyberbotics R2025a. High-level locomotion uses `from managers import RobotisOp2GaitManager, RobotisOp2MotionManager` (compiled binary shipped with Webots; may fail to import — must be behind an abstraction with a documented fallback).

## Environment facts

- Python 3.12.3. No RL libraries installed yet. `requirements.txt` lists them; the user creates the env. Webots is already installed by the user; we never install or launch it here.
- Verification runs WITHOUT Webots via a mock of the Webots `controller`/`Supervisor`/`managers` API. The env/config/robot-layer logic is tested headless.
- Keep the robot/env/config layer importable with only `numpy` + `gymnasium` (light). SB3/tensorflow are only needed by the agent files and the smoke test, which skip cleanly if those libs are absent.

## Build order (by dependency)

- [ ] 1. Create project skeleton, dependency manifest, and the Webots-API mock used by all tests.
      Create `requirements.txt` (gymnasium, stable-baselines3, tensorflow, numpy, pandas, matplotlib — unpinned or loosely pinned, documented), `models/.gitkeep`, `results/.gitkeep`, `controllers/darwin_common/__init__.py`, `tests/__init__.py`, and `tests/conftest.py` providing a `fake_webots` fixture: an in-memory mock of `controller.Supervisor`/`Robot`/`Node`/`Field`, `Accelerometer`, `Gyro`, `Motor`, plus a mock `managers` module exposing `RobotisOp2GaitManager`/`RobotisOp2MotionManager`. The mock models N named robot nodes with mutable translation/rotation, barrier nodes, a settable "tilt" for fall simulation, and a `step()` that advances a toy kinematic model (walk_forward moves +Δ along heading, turn changes heading). Also add `pytest.ini`/`pyproject` test config so `python3 -m pytest` discovers `tests/`.
      Files: `/home/juan/Documentos/parcial2 IA/requirements.txt`, `/home/juan/Documentos/parcial2 IA/models/.gitkeep`, `/home/juan/Documentos/parcial2 IA/results/.gitkeep`, `/home/juan/Documentos/parcial2 IA/controllers/darwin_common/__init__.py`, `/home/juan/Documentos/parcial2 IA/tests/__init__.py`, `/home/juan/Documentos/parcial2 IA/tests/conftest.py`, `/home/juan/Documentos/parcial2 IA/pytest.ini`
      Verify: `python3 -m pytest tests/ -q` collects 0 tests without import errors (exit 0 or "no tests ran"); `python3 -c "import tests.conftest"` works after `pip install -r requirements.txt` OR degrades to importing only the mock if deps absent.

- [ ] 2. Create `config.py` with all race constants.
      Goal line (`GOAL_Y`), per-robot start positions keyed by robot name ("DQN","ppo","actor - critic","agente propio") matching the .wbt translations, `ACTION_MAP = {0:"walk_forward",1:"turn_left",2:"turn_right"}`, `SAFETY_RADIUS`, obstacle/other-robot penalty weights, fall tilt thresholds (accelerometer/gyro), max episode steps, observation normalization bounds, progress/goal/penalty reward weights, and a parametric hook so experiments can override space size / start / goal.
      Files: `/home/juan/Documentos/parcial2 IA/controllers/darwin_common/config.py`, `/home/juan/Documentos/parcial2 IA/tests/test_config.py`
      Verify: `python3 -m pytest tests/test_config.py -q` passes (asserts action map has 3 entries, start positions exist for all 4 robot names, thresholds are positive).

- [ ] 3. Create `darwin_robot.py` — the `DarwinRobot` class wrapping the Supervisor.
      One class: `__init__` takes an injected controller backend (defaults to real Webots, mock in tests), initializes `getSelf()`, Accelerometer, Gyro, and a gait/motion manager via a `_LocomotionBackend` abstraction that tries `from managers import ...` and on Imwith ImportError/init failure falls back to a documented stub (direct motor positions / no-op with warning). Methods: `walk_forward()`, `turn_left()`, `turn_right()`, `stand()`, `get_position()`, `get_orientation()` (heading from rotation field), `is_fallen()` (torso tilt from accel/gyro vs thresholds), `reset_pose(x,y)` (set translation/rotation via Supervisor fields), `tick()` (one `step()`), `find_others()` / `find_barriers()` (scan root children by `name`, excluding self). Do NOT rely on `getFromDef("DARWIN")`.
      Files: `/home/juan/Documentos/parcial2 IA/controllers/darwin_common/darwin_robot.py`, `/home/juan/Documentos/parcial2 IA/tests/test_darwin_robot.py`
      Verify: `python3 -m pytest tests/test_darwin_robot.py -q` passes (walk moves position, turn changes heading, tilt beyond threshold -> is_fallen True, reset_pose sets translation, locomotion fallback path triggers on mock ImportError without raising).

- [ ] 4. Create `race_env.py` — `DarwinRaceEnv(gymnasium.Env)`.
      `observation_space = Box` of 5 normalized floats [dist to goal, angle to goal, heading, dist to nearest obstacle, dist to nearest other robot]; `action_space = Discrete(3)`. `reset()` resets pose from config and returns obs+info. `step(a)` maps action via DarwinRobot, ticks, computes reward (+progress toward goal, +large bonus and `terminated` on crossing goal, −penalty ∝ intrusion into safety radius of obstacles/other robots, −penalty and `terminated` on fall or hard collision), enforces max-steps `truncated`, and fills `info` with safety events (`fall`, `collision`, `margin_violations`, `progress`, `success`). Constructor accepts overrides for space size / start / goal (for experiments).
      Files: `/home/juan/Documentos/parcial2 IA/controllers/darwin_common/race_env.py`, `/home/juan/Documentos/parcial2 IA/tests/test_race_env.py`
      Verify: `python3 -m pytest tests/test_race_env.py -q` passes — asserts obs shape/space, reward sign on forward progress, goal bonus + terminated on crossing, fall -> penalty+terminated, obstacle intrusion -> negative penalty and info flag, truncation at max steps. Plus `from stable_baselines3.common.env_checker import check_env; check_env(DarwinRaceEnv(backend=mock))` runs clean (test skips with `pytest.importorskip("stable_baselines3")` if SB3 absent).

- [ ] 5. Create `agent_runner.py` — shared train/load/race helper.
      Functions: `make_env(robot_name, backend, **overrides)`; `train(algo_cls, env, timesteps, save_path, **hp)` -> trains an SB3 model, saves `.zip`; `load(algo_cls, path)`; `run_race(model, env, deterministic=True)` -> runs one episode returning metrics dict (success, steps, cumulative_reward, falls, collisions). Reads mode ("train"/"race") from CLI arg or `MODE` env var. Keeps hyperparameters default-ish and documented.
      Files: `/home/juan/Documentos/parcial2 IA/controllers/darwin_common/agent_runner.py`, `/home/juan/Documentos/parcial2 IA/tests/test_agent_runner.py`
      Verify: `python3 -m pytest tests/test_agent_runner.py -q` passes — mode parsing from arg/env; `run_race` returns the metrics keys against the mock env (SB3-dependent parts guarded by `importorskip`).

- [ ] 6. Create the 3 SB3 agent controllers (one file each, same pattern).
      Each reads mode, builds env for its robot via `agent_runner`, and in train mode fits the SB3 algorithm with `MlpPolicy` saving to `models/<algo>.zip`; in race mode loads and runs deterministically. ppo_agent -> PPO, robot "ppo", `models/ppo.zip`; dqn_agent -> DQN, robot "DQN", `models/dqn.zip`; a2c_agent -> A2C, robot "actor - critic", `models/a2c.zip`. Controller folder names: `ppo_agent`, `dqn_agent`, `a2c_agent`.
      Files: `/home/juan/Documentos/parcial2 IA/controllers/ppo_agent/ppo_agent.py`, `/home/juan/Documentos/parcial2 IA/controllers/dqn_agent/dqn_agent.py`, `/home/juan/Documentos/parcial2 IA/controllers/a2c_agent/a2c_agent.py`, `/home/juan/Documentos/parcial2 IA/tests/test_sb3_agents_smoke.py`
      Verify: `python3 -m pytest tests/test_sb3_agents_smoke.py -q` — for each algo, a few timesteps of training against the mock env produces a reloadable `.zip` under a tmp dir (test uses `importorskip("stable_baselines3")`; skips if absent). Also `python3 -m py_compile controllers/ppo_agent/ppo_agent.py controllers/dqn_agent/dqn_agent.py controllers/a2c_agent/a2c_agent.py`.

- [ ] 7. Create the experiments script.
      `run_experiments.py` runs >=20 configs varying space size, start position, goal position, and per-algorithm hyperparameters; logs per-config metrics (cumulative_reward, success_rate, steps/time to goal, falls, collisions) to `results/experiments.csv` via pandas. Supports a `--quick` reduced-config mode for CI/verification. Uses the mock backend when Webots is unavailable so it is runnable headless.
      Files: `/home/juan/Documentos/parcial2 IA/experiments/run_experiments.py`, `/home/juan/Documentos/parcial2 IA/experiments/__init__.py`, `/home/juan/Documentos/parcial2 IA/tests/test_experiments.py`
      Verify: `python3 -m pytest tests/test_experiments.py -q` passes — running the quick mode produces a CSV with the expected columns and >=1 row (full run asserted to enumerate >=20 configs via the config generator, tested directly without executing all 20).

- [ ] 8. Create the plotting script.
      `plot_results.py` reads `results/experiments.csv`, builds summary tables (per-algorithm averages) and comparison figures (cumulative reward, success rate) as PNGs in `results/`. Uses matplotlib `Agg` backend.
      Files: `/home/juan/Documentos/parcial2 IA/experiments/plot_results.py`, `/home/juan/Documentos/parcial2 IA/tests/test_plots.py`
      Verify: `python3 -m pytest tests/test_plots.py -q` passes — given a small synthetic CSV, it writes the expected PNG files and a summary table CSV/console output (uses `importorskip("matplotlib")`).

- [ ] 9. Create the custom agent (LAST of the code).
      `custom_agent.py` for robot "agente propio" implements a MANUAL learning/decision loop (NO stable-baselines3) reusing `DarwinRaceEnv` — a clean, documented discretized tabular Q-learning baseline over the observation, clearly marked "replace with Taller-2 algorithm" with a TODO and an explicit extension point. Integrate it into `run_experiments.py` and `plot_results.py` as a 4th algorithm. Controller folder `custom_agent`.
      Files: `/home/juan/Documentos/parcial2 IA/controllers/custom_agent/custom_agent.py`, `/home/juan/Documentos/parcial2 IA/tests/test_custom_agent.py`, (edit) `/home/juan/Documentos/parcial2 IA/experiments/run_experiments.py`, `/home/juan/Documentos/parcial2 IA/experiments/plot_results.py`
      Verify: `python3 -m pytest tests/test_custom_agent.py -q` passes — tabular agent trains a few episodes on the mock env, Q-table updates, `run_race` returns metrics; experiments quick mode now includes the custom algorithm row.

- [ ] 10. Final Webots integration (LAST). Separate from code; do not modify track/robot geometry.
      In `pista_atletismo_50m.wbt` set `controller "<none>"` -> folder name per robot: DQN->`"dqn_agent"`, ppo->`"ppo_agent"`, actor - critic->`"a2c_agent"`, agente propio->`"custom_agent"`, changing ONLY the `controller` field of each robot block. Add a race-mode entry point (in `agent_runner`/a small `race.py`) that detects which robot crosses the finish line (y=GOAL_Y) first and reports the winner. Write a `README.md` documenting exactly what the user runs inside Webots (set MODE=train per controller, then MODE=race).
      Files: (edit) `/home/juan/Documentos/parcial2 IA/pista_atletismo_50m.wbt`, `/home/juan/Documentos/parcial2 IA/README.md`, optionally `/home/juan/Documentos/parcial2 IA/controllers/darwin_common/race.py`
      Verify: `python3 -m pytest tests/ -q` — full suite green; `grep` confirms exactly 4 controller-field changes and no geometry change via `git diff --stat pista_atletismo_50m.wbt` showing only controller lines; `python3 -c "..."` parse check that each robot name maps to its folder.

## How verification runs (for the reviewer)

- Full suite: `cd "/home/juan/Documentos/parcial2 IA" && python3 -m pytest tests/ -q`.
- `check_env`: inside `tests/test_race_env.py` via `stable_baselines3.common.env_checker.check_env` against the mock backend.
- Smoke test: `tests/test_sb3_agents_smoke.py` trains each SB3 agent a handful of timesteps and reloads the `.zip`.
- Experiments/plots: `tests/test_experiments.py` + `tests/test_plots.py` run quick mode and assert CSV columns and PNG outputs.
- SB3/tensorflow/matplotlib-dependent tests use `pytest.importorskip` so the core env/config/robot tests pass even before the user installs the full requirements. Webots is never launched.

## Assumptions / gaps noted

- Taller-2 algorithm is unspecified -> implemented as a documented tabular Q-learning placeholder with a clear replacement point (per task instructions).
- Race direction is along the y axis (finish line at y=9), inferred from START_FINISH geometry and robot start y-coordinates.
- Robot lookup is by `name` (not DEF) because all 4 robots share `DEF DARWIN`.
