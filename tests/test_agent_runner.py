"""Tests del helper compartido agent_runner.

Cubre:
- Parseo del modo (train/race) desde argumento de CLI y desde la variable de
  entorno MODE.
- make_env construye un DarwinRaceEnv sobre el backend mock.
- run_race devuelve las claves de métricas esperadas usando una política
  dummy (no requiere SB3).

Las partes que dependen de stable-baselines3 se cubren en el smoke test aparte.
"""

import agent_runner
from race_env import DarwinRaceEnv


def test_get_mode_from_cli_arg():
    assert agent_runner.get_mode(["train"]) == "train"
    assert agent_runner.get_mode(["race"]) == "race"
    # Acepta otros args de Webots mezclados; detecta el modo válido.
    assert agent_runner.get_mode(["--foo", "TRAIN", "bar"]) == "train"


def test_get_mode_from_env(monkeypatch):
    monkeypatch.setenv("MODE", "train")
    assert agent_runner.get_mode([]) == "train"
    monkeypatch.setenv("MODE", "race")
    assert agent_runner.get_mode([]) == "race"


def test_get_mode_cli_takes_priority_over_env(monkeypatch):
    monkeypatch.setenv("MODE", "race")
    assert agent_runner.get_mode(["train"]) == "train"


def test_get_mode_default_when_absent(monkeypatch):
    monkeypatch.delenv("MODE", raising=False)
    assert agent_runner.get_mode([]) == "race"
    assert agent_runner.get_mode([], default="train") == "train"
    # Valor inválido cae al default.
    monkeypatch.setenv("MODE", "garbage")
    assert agent_runner.get_mode([]) == "race"


def test_make_env_builds_race_env(fake_webots):
    backend = fake_webots.make_supervisor(self_name="ppo")
    env = agent_runner.make_env("ppo", backend)
    assert isinstance(env, DarwinRaceEnv)
    assert env.robot_name == "ppo"
    assert env.action_space.n == 3


def test_make_env_passes_overrides(fake_webots):
    backend = fake_webots.make_supervisor(self_name="ppo")
    env = agent_runner.make_env("ppo", backend, goal_y=7.9)
    assert env.params.goal_y == 7.9


class _DummyModel:
    """Política mínima: siempre devuelve una acción fija (no usa SB3)."""

    def __init__(self, action=0):
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, None


def test_run_race_returns_metric_keys(fake_webots):
    backend = fake_webots.make_supervisor(self_name="ppo")
    # Pocos pasos para que el episodio termine por truncation sin completar la
    # vuelta (el éxito ahora exige recorrer cp1..cp7 y volver a cp0).
    env = agent_runner.make_env("ppo", backend)
    env.params = env.params.override(max_episode_steps=5)
    metrics = agent_runner.run_race(_DummyModel(action=0), env)
    assert set(metrics) == {
        "success", "steps", "cumulative_reward", "falls", "collisions",
        "checkpoints_reached",
    }
    assert isinstance(metrics["success"], bool)
    assert isinstance(metrics["steps"], int)
    assert isinstance(metrics["cumulative_reward"], float)
    assert isinstance(metrics["falls"], int)
    assert isinstance(metrics["collisions"], int)
    assert isinstance(metrics["checkpoints_reached"], int)
    assert metrics["steps"] >= 1


def test_run_race_truncates_without_completing_lap(fake_webots):
    backend = fake_webots.make_supervisor(self_name="ppo")
    # Pocos pasos girando: no completa la vuelta -> success False, truncation.
    env = agent_runner.make_env("ppo", backend)
    env.params = env.params.override(max_episode_steps=5)
    metrics = agent_runner.run_race(_DummyModel(action=1), env)  # solo gira
    assert metrics["success"] is False
    assert metrics["steps"] == 5
