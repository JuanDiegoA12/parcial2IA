"""Controlador Webots del robot 'DQN': agente DQN (stable-baselines3).

DQN (Deep Q-Network) es un algoritmo off-policy basado en valor: aprende una
Q-función con una red MLP y un replay buffer, adecuado para espacios de acción
discretos como el nuestro (Discrete(3)).

Dos modos (ver agent_runner.get_mode):
    train -> entrena una política MlpPolicy y guarda models/dqn.zip.
    race  -> recarga models/dqn.zip y corre un episodio de forma determinista.

Hiperparámetros: valores por defecto de SB3 (documentados abajo) salvo la
semilla; la sofisticación se reserva para el agente propio.
"""

from __future__ import annotations

import os
import sys

# Permite importar el paquete común (agent_runner, race_env) tanto bajo el
# pythonpath de pytest como al ejecutarse como controlador de Webots.
_COMMON = os.path.join(os.path.dirname(__file__), "..", "darwin_common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import agent_runner

ROBOT_NAME = "DQN"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "dqn.zip")

# Pasos de entrenamiento por defecto (se puede ajustar con TIMESTEPS en el env).
TRAIN_TIMESTEPS = int(os.environ.get("TIMESTEPS", "100000"))

# Hiperparámetros default-ish de DQN (valores por defecto de SB3 salvo seed):
#   learning_rate=1e-4, buffer_size=1e6, learning_starts=100,
#   batch_size=32, gamma=0.99, train_freq=4, target_update_interval=1e4,
#   exploration_fraction=0.1. Solo fijamos la semilla para reproducibilidad.
HYPERPARAMS = {
    "seed": 0,
    "verbose": 1,
}


def _build_backend():
    """Obtiene el Supervisor de Webots (solo disponible dentro del simulador)."""
    from controller import Supervisor

    return Supervisor()


def main():
    mode = agent_runner.get_mode()
    backend = _build_backend()
    env = agent_runner.make_env(ROBOT_NAME, backend)

    if mode == "train":
        from stable_baselines3 import DQN

        agent_runner.train(DQN, env, TRAIN_TIMESTEPS, MODEL_PATH, **HYPERPARAMS)
    else:
        from stable_baselines3 import DQN

        model = agent_runner.load(DQN, MODEL_PATH)
        metrics = agent_runner.run_race(model, env)
        print("[dqn] métricas de carrera:", metrics)


if __name__ == "__main__":
    main()
