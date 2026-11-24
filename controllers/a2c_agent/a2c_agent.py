"""Controlador Webots del robot 'actor - critic': agente A2C (stable-baselines3).

A2C (Advantage Actor-Critic) es un algoritmo on-policy actor-crítico: una red
estima la política (actor) y otra el valor (crítico), usando la ventaja para
reducir la varianza del gradiente. Es la versión síncrona del clásico A3C.

Nota de distribución: la carpeta del controlador es `a2c_agent`, así que el
campo 'controller' de este robot en el .wbt será 'a2c_agent' (aunque el robot
se llame 'actor - critic').

Dos modos (ver agent_runner.get_mode):
    train -> entrena una política MlpPolicy y guarda models/a2c.zip.
    race  -> recarga models/a2c.zip y corre un episodio de forma determinista.

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

ROBOT_NAME = "actor - critic"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "a2c.zip")

# Pasos de entrenamiento por defecto (se puede ajustar con TIMESTEPS en el env).
TRAIN_TIMESTEPS = int(os.environ.get("TIMESTEPS", "100000"))

# Hiperparámetros default-ish de A2C (valores por defecto de SB3 salvo seed):
#   learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=1.0,
#   ent_coef=0.0, vf_coef=0.5. Solo fijamos la semilla para reproducibilidad.
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
        from stable_baselines3 import A2C

        agent_runner.train(A2C, env, TRAIN_TIMESTEPS, MODEL_PATH, **HYPERPARAMS)
    else:
        from stable_baselines3 import A2C

        model = agent_runner.load(A2C, MODEL_PATH)
        metrics = agent_runner.run_race(model, env)
        print("[a2c] métricas de carrera:", metrics)


if __name__ == "__main__":
    main()
