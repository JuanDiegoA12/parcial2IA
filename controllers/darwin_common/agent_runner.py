"""Helper compartido para los agentes stable-baselines3 (PPO, DQN, A2C).

Centraliza lo que los tres controladores hacen igual: construir el entorno de
carrera, entrenar un modelo SB3, recargarlo, correr un episodio de carrera y
leer el modo de ejecución (train/race). Así cada controlador queda en un solo
archivo pequeño que solo fija su algoritmo, nombre de robot y ruta del modelo.

Diseño: stable-baselines3 se importa de forma perezosa DENTRO de las funciones
que lo necesitan. Importar este módulo NO requiere tener instalado SB3, de modo
que los tests headless pueden importarlo con solo numpy+gymnasium.

El backend de Webots (Supervisor real o su mock) siempre se inyecta: este
módulo nunca importa `controller` ni lanza el simulador.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

try:  # entorno como módulo del paquete o como módulo suelto (pythonpath).
    from .race_env import DarwinRaceEnv
except ImportError:  # pragma: no cover - ruta usada bajo pythonpath plano
    from race_env import DarwinRaceEnv


def make_env(robot_name: str, backend, **overrides) -> DarwinRaceEnv:
    """Construye un DarwinRaceEnv para `robot_name` sobre el `backend` dado.

    Parameters
    ----------
    robot_name:
        Nombre exacto del robot en el .wbt ('ppo', 'DQN', 'actor - critic').
    backend:
        Supervisor de Webots real o su mock (fake_webots.make_supervisor(...)).
    **overrides:
        Se pasan tal cual a DarwinRaceEnv (p. ej. goal_y, start_position,
        params) para variar meta/inicio/espacio en los experimentos.
    """
    return DarwinRaceEnv(backend, robot_name=robot_name, **overrides)


def train(algo_cls, env, timesteps: int, save_path: str, **hp):
    """Entrena un modelo SB3 con política MLP y lo guarda como .zip.

    Parameters
    ----------
    algo_cls:
        Clase del algoritmo SB3 (stable_baselines3.PPO/DQN/A2C).
    env:
        Entorno gymnasium ya construido (ver make_env).
    timesteps:
        Número de pasos de entrenamiento (model.learn(total_timesteps=...)).
    save_path:
        Ruta destino del .zip (SB3 añade la extensión si falta).
    **hp:
        Hiperparámetros extra para el constructor del algoritmo
        (learning_rate, gamma, etc.). Se mantienen default-ish.

    Returns
    -------
    El modelo SB3 entrenado.
    """
    # Import perezoso: solo se exige SB3 cuando realmente se entrena.
    model = algo_cls("MlpPolicy", env, **hp)
    model.learn(total_timesteps=timesteps)
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    model.save(save_path)
    return model


def load(algo_cls, path: str):
    """Recarga un modelo SB3 previamente guardado desde `path` (.zip)."""
    return algo_cls.load(path)


def run_race(model, env, deterministic: bool = True) -> Dict[str, object]:
    """Corre un episodio completo con `model` sobre `env` y devuelve métricas.

    Returns
    -------
    dict con las claves:
        success (bool): el robot cruzó la meta.
        steps (int): pasos ejecutados en el episodio.
        cumulative_reward (float): recompensa acumulada.
        falls (int): número de caídas detectadas.
        collisions (int): número de colisiones duras detectadas.
    """
    obs, _info = env.reset()
    cumulative_reward = 0.0
    steps = 0
    falls = 0
    collisions = 0
    success = False

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        steps += 1
        if info.get("fall"):
            falls += 1
        if info.get("collision"):
            collisions += 1
        if info.get("success"):
            success = True

    return {
        "success": bool(success),
        "steps": int(steps),
        "cumulative_reward": float(cumulative_reward),
        "falls": int(falls),
        "collisions": int(collisions),
    }


def get_mode(argv: Optional[list] = None, default: str = "race") -> str:
    """Lee el modo de ejecución ('train' o 'race').

    Prioridad: primer argumento de línea de comandos > variable de entorno
    MODE > `default`. Solo se aceptan 'train' y 'race'; cualquier otro valor
    cae en `default`.

    Parameters
    ----------
    argv:
        Lista de argumentos (por defecto sys.argv). Se busca 'train'/'race'
        entre los argumentos (Webots inyecta otros args al controlador).
    default:
        Modo a usar si no se indica ninguno.
    """
    if argv is None:
        argv = sys.argv[1:]

    for arg in argv:
        value = str(arg).strip().lower()
        if value in ("train", "race"):
            return value

    env_mode = os.environ.get("MODE")
    if env_mode is not None:
        value = env_mode.strip().lower()
        if value in ("train", "race"):
            return value

    return default
