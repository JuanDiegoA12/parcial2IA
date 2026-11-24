"""Tests de DarwinRaceEnv contra el backend mock (fake_webots)."""

import os

import numpy as np
import pytest

import config
from race_env import DarwinRaceEnv


@pytest.fixture(autouse=True)
def _restore_locomotion_mode():
    """Restaura LOCOMOTION_MODE tras cada test para no filtrar el modo."""
    saved = os.environ.get("LOCOMOTION_MODE")
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("LOCOMOTION_MODE", None)
        else:
            os.environ["LOCOMOTION_MODE"] = saved


def _make_env(fake_webots, name="ppo", mode=None, **kwargs):
    # El modo de locomoción se resuelve al construir el robot (dentro del env),
    # así que debe fijarse ANTES de instanciar DarwinRaceEnv.
    if mode is not None:
        os.environ["LOCOMOTION_MODE"] = mode
    supervisor = fake_webots.make_supervisor(self_name=name)
    return DarwinRaceEnv(supervisor, robot_name=name, **kwargs), supervisor


def test_spaces_shape(fake_webots):
    env, _ = _make_env(fake_webots)
    assert env.action_space.n == 3
    assert env.observation_space.shape == (5,)


def test_reset_returns_obs_and_info(fake_webots):
    env, _ = _make_env(fake_webots, "ppo")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray) and obs.shape == (5,)
    assert env.observation_space.contains(obs)
    assert info["robot_name"] == "ppo"


def test_forward_progress_is_positive(fake_webots):
    # ppo arranca en y=7.88, meta en y=9 -> caminar reduce la distancia.
    env, _ = _make_env(fake_webots, "ppo")
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # walk_forward
    assert info["progress"] > 0
    assert reward > 0
    assert not terminated


def test_crossing_goal_grants_bonus_and_terminates(fake_webots):
    # Meta justo por debajo del destino de un paso para cruzarla en un tick.
    # Se deriva del paso cinemático real (config.KINEMATIC_LINEAR_STEP, fuente
    # única de la velocidad) en vez de un literal, para no romperse al afinar la
    # velocidad: ppo arranca en y=7.88 y avanza +KINEMATIC_LINEAR_STEP en +y.
    goal_y = 7.88 + config.KINEMATIC_LINEAR_STEP / 2.0
    env, _ = _make_env(fake_webots, "ppo", goal_y=goal_y)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["success"] is True
    assert terminated is True
    assert reward > env.params.goal_bonus - 10


def test_obstacle_intrusion_penalizes(fake_webots):
    # Colocamos al robot junto a una valla y comprobamos penalización.
    env, supervisor = _make_env(fake_webots, "ppo")
    env.reset()
    # 'valla 1(9)' está en (-0.895, -5.974). Reubicamos al robot cerca.
    barrier = fake_webots.node("valla 1(9)").translation
    supervisor.getSelf().set_translation([barrier[0] + 0.1, barrier[1] + 0.1, 0.34])
    env._prev_dist = env._dist_to_goal()
    obs, reward, terminated, truncated, info = env.step(1)  # turn, no avanza
    assert info["margin_violations"] >= 1
    assert reward < 0


def test_fall_penalizes_and_terminates(fake_webots):
    # La caída sólo se detecta en modo gait (Plan A); kinematic la desactiva.
    env, supervisor = _make_env(fake_webots, "ppo", mode="gait")
    env.reset()
    supervisor.force_tilt(config.FALL_TILT_THRESHOLD + 0.5)
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["fall"] is True
    assert terminated is True
    assert reward < 0


def test_truncation_at_max_steps(fake_webots):
    env, _ = _make_env(fake_webots, "ppo", goal_y=1000.0)
    env.reset()
    env.params = env.params.override(max_episode_steps=3)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(1)  # girar: nunca llega
        if terminated:
            break
    assert truncated is True


def test_check_env_sb3(fake_webots):
    sb3_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    env, _ = _make_env(fake_webots, "ppo")
    # check_env valida espacios, tipos y el contrato reset/step.
    sb3_checker.check_env(env, warn=True, skip_render_check=True)


@pytest.mark.parametrize("mode", ["gait", "kinematic"])
def test_check_env_both_modes(fake_webots, mode):
    # El contrato del env SB3 debe cumplirse en AMBOS modos de locomoción.
    sb3_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    env, _ = _make_env(fake_webots, "ppo", mode=mode)
    sb3_checker.check_env(env, warn=True, skip_render_check=True)


def test_obstacle_intrusion_penalizes_in_kinematic_mode(fake_webots):
    # Las penalizaciones por proximidad a obstáculos siguen activas en modo
    # kinematic (Plan B): sólo la caída se desactiva, no las colisiones.
    env, supervisor = _make_env(fake_webots, "ppo", mode="kinematic")
    assert env.robot.locomotion.mode == "kinematic"
    env.reset()
    barrier = fake_webots.node("valla 1(9)").translation
    supervisor.getSelf().set_translation([barrier[0] + 0.1, barrier[1] + 0.1, 0.34])
    env._prev_dist = env._dist_to_goal()
    obs, reward, terminated, truncated, info = env.step(1)  # girar, no avanza
    assert info["margin_violations"] >= 1
    assert reward < 0
