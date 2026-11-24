"""Controlador Webots del robot 'ppo': agente PPO (stable-baselines3).

PPO (Proximal Policy Optimization) es un algoritmo on-policy de gradiente de
política con recorte (clipping) que equilibra estabilidad y simplicidad; es la
opción por defecto recomendada por SB3 para espacios continuos/discretos.

Dos modos (ver agent_runner.get_mode):
    train -> entrena una política MlpPolicy y guarda models/ppo.zip.
    race  -> recarga models/ppo.zip y corre un episodio de forma determinista.

Hiperparámetros: se dejan en los valores por defecto de SB3 (documentados
abajo) salvo la semilla; la sofisticación se reserva para el agente propio.
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

ROBOT_NAME = "ppo"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo.zip")

# Pasos de entrenamiento por defecto (se puede ajustar con TIMESTEPS en el env).
TRAIN_TIMESTEPS = int(os.environ.get("TIMESTEPS", "100000"))

# Hiperparámetros default-ish de PPO (valores por defecto de SB3 salvo seed):
#   learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99,
#   gae_lambda=0.95, clip_range=0.2. Solo fijamos la semilla para
#   reproducibilidad.
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
        from stable_baselines3 import PPO

        agent_runner.train(PPO, env, TRAIN_TIMESTEPS, MODEL_PATH, **HYPERPARAMS)
    else:
        from stable_baselines3 import PPO

        model = agent_runner.load(PPO, MODEL_PATH)
        metrics = agent_runner.run_race(model, env)
        print("[ppo] métricas de carrera:", metrics)


if __name__ == "__main__":
    main()
