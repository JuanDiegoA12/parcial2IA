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
    # La observación creció a 7 dims (dirección/distancia al SIGUIENTE
    # checkpoint + progreso de la vuelta + proximidad a obstáculos/robots).
    assert env.observation_space.shape == (7,)


def test_reset_returns_obs_and_info(fake_webots):
    env, _ = _make_env(fake_webots, "ppo")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray) and obs.shape == (7,)
    assert env.observation_space.contains(obs)
    assert info["robot_name"] == "ppo"


def test_forward_progress_toward_next_checkpoint_is_positive(fake_webots):
    # El primer objetivo es cp1 (8.79, 7.98). Colocamos al robot SOBRE la línea
    # de carrera cp0->cp1, cerca de cp1 (desviación off-track ~0), orientado
    # hacia cp1, y damos un paso de avance: la distancia al objetivo baja, así
    # que el progreso y la recompensa son positivos.
    import math

    env, supervisor = _make_env(fake_webots, "ppo", mode="kinematic")
    env.reset()
    node = supervisor.getSelf()

    cp0 = (0.0, config.GOAL_Y)
    cp1 = (8.79, 7.98)
    # Punto sobre el segmento cp0->cp1 al 90% del camino (sobre la línea).
    t = 0.9
    px = cp0[0] + t * (cp1[0] - cp0[0])
    py = cp0[1] + t * (cp1[1] - cp0[1])
    # Heading que hace que el avance (-sin θ, cos θ) apunte hacia cp1.
    dx, dy = cp1[0] - px, cp1[1] - py
    theta = math.atan2(-dx, dy)
    node.set_translation([px, py, 0.34])
    node.set_rotation([0.0, 0.0, 1.0, theta])
    env.tracker.prev_dist = env._dist_to_target()

    obs, reward, terminated, truncated, info = env.step(0)  # walk_forward
    assert info["progress"] > 0
    assert reward > 0
    assert not terminated


def test_completing_full_lap_wins_and_terminates(fake_webots):
    # Conducir la vuelta teletransportando el nodo a cada objetivo en orden y
    # pisando step() para que el tracker avance. Sólo el RETORNO final a cp0
    # (tras cp1..cp7) debe marcar success/terminated.
    env, supervisor = _make_env(fake_webots, "ppo", mode="kinematic")
    env.reset()
    node = supervisor.getSelf()
    targets = env.tracker.targets  # [cp1..cp7, cp0]
    last_info = None
    for i, (tx, ty) in enumerate(targets):
        node.set_translation([tx, ty, 0.34])
        _, _, terminated, _, info = env.step(1)  # girar (no mueve en kinematic)
        last_info = info
        if i < len(targets) - 1:
            # Antes del cierre de la vuelta NO hay victoria.
            assert info["success"] is False
            assert terminated is False
            assert info["checkpoints_reached"] == i + 1
    assert last_info["success"] is True
    assert last_info["checkpoints_reached"] == config.TOTAL_TARGETS


def test_crossing_goal_without_lap_does_not_win(fake_webots):
    # Cruzar y=GOAL_Y (estar en (0,9)) SIN haber recorrido cp1..cp7 NO gana.
    env, supervisor = _make_env(fake_webots, "ppo", mode="kinematic")
    env.reset()
    node = supervisor.getSelf()
    node.set_translation([0.0, config.GOAL_Y, 0.34])  # sobre la meta cp0
    env.tracker.prev_dist = env._dist_to_target()
    _, _, terminated, _, info = env.step(1)
    assert info["success"] is False
    assert terminated is False
    assert info["checkpoints_reached"] == 0


def test_obstacle_intrusion_penalizes(fake_webots):
    # Colocamos al robot junto a una valla y comprobamos penalización.
    env, supervisor = _make_env(fake_webots, "ppo")
    env.reset()
    # 'valla 1(9)' está en (-0.895, -5.974). Reubicamos al robot cerca.
    barrier = fake_webots.node("valla 1(9)").translation
    supervisor.getSelf().set_translation([barrier[0] + 0.1, barrier[1] + 0.1, 0.34])
    env.tracker.prev_dist = env._dist_to_target()
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
    env.tracker.prev_dist = env._dist_to_target()
    obs, reward, terminated, truncated, info = env.step(1)  # girar, no avanza
    assert info["margin_violations"] >= 1
    assert reward < 0
