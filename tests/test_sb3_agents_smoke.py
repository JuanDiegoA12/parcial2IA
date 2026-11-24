"""Smoke test de los agentes SB3 (PPO, DQN, A2C) contra el entorno mock.

Para cada algoritmo: entrena un puñado de timesteps sobre el entorno mock,
guarda un .zip en un tmp path y verifica que se puede recargar. Todo el test se
salta limpiamente si stable-baselines3 no está instalado (pytest.importorskip).

El entrenamiento es de muy pocos pasos: solo comprueba el contrato de
train/load del runner, no la calidad de la política.
"""

import pytest

pytest.importorskip("stable_baselines3")

import agent_runner  # noqa: E402  (tras el importorskip)


def _train_and_reload(algo_cls, robot_name, fake_webots, tmp_path, timesteps):
    backend = fake_webots.make_supervisor(self_name=robot_name)
    env = agent_runner.make_env(robot_name, backend)
    save_path = str(tmp_path / f"{robot_name}.zip")
    agent_runner.train(algo_cls, env, timesteps, save_path, seed=0)
    assert (tmp_path / f"{robot_name}.zip").exists()
    model = agent_runner.load(algo_cls, save_path)
    # El modelo recargado predice una acción válida del espacio.
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))


def test_ppo_smoke(fake_webots, tmp_path):
    from stable_baselines3 import PPO

    _train_and_reload(PPO, "ppo", fake_webots, tmp_path, timesteps=64)


def test_dqn_smoke(fake_webots, tmp_path):
    from stable_baselines3 import DQN

    _train_and_reload(DQN, "DQN", fake_webots, tmp_path, timesteps=64)


def test_a2c_smoke(fake_webots, tmp_path):
    from stable_baselines3 import A2C

    _train_and_reload(A2C, "actor - critic", fake_webots, tmp_path, timesteps=64)
