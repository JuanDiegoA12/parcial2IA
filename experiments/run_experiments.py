"""Generador y ejecutor de experimentos de la carrera DARwIn-OP.

Enumera >=20 configuraciones que varían cuatro dimensiones:
  1. tamaño del espacio (space_size -> cota de normalización max_dist),
  2. posición inicial del robot (start),
  3. posición de la meta (goal),
  4. hiperparámetros por algoritmo (PPO / DQN / A2C).

Para cada configuración entrena brevemente el agente SB3 correspondiente contra
el backend mock (cuando Webots no está disponible, que es siempre fuera del
simulador) y corre un episodio de carrera, registrando las métricas en
results/experiments.csv vía pandas.

Uso:
    python3 experiments/run_experiments.py            # set completo
    python3 experiments/run_experiments.py --quick    # set reducido, timesteps mínimos

El generador de configuraciones `generate_configs()` es importable para que un
test pueda afirmar que produce >=20 configuraciones sin ejecutarlas todas.

stable-baselines3 se importa de forma perezosa dentro de run_config(); importar
este módulo y llamar a generate_configs() NO requiere SB3 (ni Webots).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- Resolución de rutas de importación -------------------------------------
# Permite ejecutar este script directamente (python3 experiments/run_experiments.py)
# resolviendo los módulos del paquete común igual que bajo pytest.ini.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "controllers"),
           os.path.join(_ROOT, "controllers", "darwin_common"),
           os.path.join(_ROOT, "controllers", "custom_agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as race_config  # noqa: E402  (tras ajustar sys.path)

RESULTS_DIR = os.path.join(_ROOT, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "experiments.csv")

# Columnas de results/experiments.csv (esquema estable para plot_results y FEAT-004).
CSV_COLUMNS: List[str] = [
    "algo",
    "config_id",
    "space_size",
    "start",
    "goal",
    "cumulative_reward",
    "success",
    "steps_to_goal",
    "falls",
    "collisions",
]

# Algoritmos cubiertos: los tres base (SB3) más el agente propio (CUSTOM), que
# es Q-learning tabular manual (ver controllers/custom_agent/custom_agent.py).
ALGOS: Tuple[str, ...] = ("PPO", "DQN", "A2C", "CUSTOM")

# Nombre exacto del robot en el .wbt por algoritmo (ver config.START_POSITIONS).
ROBOT_BY_ALGO: Dict[str, str] = {
    "PPO": "ppo",
    "DQN": "DQN",
    "A2C": "actor - critic",
    "CUSTOM": "agente propio",
}

# Conjunto de algoritmos SB3 (el resto se entrena con su propio trainer).
SB3_ALGOS: Tuple[str, ...] = ("PPO", "DQN", "A2C")

# Rejillas de variación por dimensión -----------------------------------------
# Tamaño del espacio: cota de normalización max_dist (m). Distintos "tamaños"
# de pista para probar robustez de la observación normalizada.
SPACE_SIZES: Tuple[float, ...] = (30.0, 60.0, 90.0)

# Posiciones iniciales a probar (x, y, z). La primera es None = usar la de config.
START_POSITIONS: Tuple[Optional[Tuple[float, float, float]], ...] = (
    None,
    (0.0, 6.0, 0.34),
    (0.5, 7.0, 0.34),
)

# Posiciones de meta (plano y). Distintas distancias a recorrer.
GOALS: Tuple[float, ...] = (8.0, 9.0, 10.0)

# Hiperparámetros por algoritmo (default-ish, documentados). Cada entrada es un
# dict pasado al constructor SB3; se mantienen simples a propósito.
HYPERPARAMS: Dict[str, Tuple[Dict[str, object], ...]] = {
    # PPO: variamos learning_rate y gamma (resto por defecto de SB3).
    "PPO": (
        {"learning_rate": 3e-4, "gamma": 0.99},
        {"learning_rate": 1e-3, "gamma": 0.95},
    ),
    # DQN: variamos learning_rate y exploración final.
    "DQN": (
        {"learning_rate": 1e-3, "exploration_final_eps": 0.05},
        {"learning_rate": 5e-4, "exploration_final_eps": 0.02},
    ),
    # A2C: variamos learning_rate y gamma.
    "A2C": (
        {"learning_rate": 7e-4, "gamma": 0.99},
        {"learning_rate": 1e-3, "gamma": 0.98},
    ),
    # CUSTOM (agente propio, Q-learning tabular): variamos alpha y epsilon_decay.
    "CUSTOM": (
        {"alpha": 0.1, "epsilon_decay": 0.99},
        {"alpha": 0.2, "epsilon_decay": 0.95},
    ),
}


@dataclass
class ExperimentConfig:
    """Una configuración concreta de experimento."""

    config_id: str
    algo: str
    robot_name: str
    space_size: float
    start: Optional[Tuple[float, float, float]]
    goal: float
    hyperparams: Dict[str, object] = field(default_factory=dict)


def generate_configs() -> List[ExperimentConfig]:
    """Enumera todas las configuraciones del experimento completo.

    Combina, por cada algoritmo, el producto cartesiano de tamaño de espacio,
    posición inicial, meta e hiperparámetros, aplicando una rejilla reducida
    (una variación por dimensión y algoritmo) para no explotar el total pero
    garantizando >=20 configuraciones distintas que varían las cuatro
    dimensiones requeridas.

    Es importable y puro (no entrena ni toca Webots) para que los tests puedan
    contar las configuraciones sin ejecutarlas.
    """
    configs: List[ExperimentConfig] = []
    for algo in ALGOS:
        robot = ROBOT_BY_ALGO[algo]
        hp_options = HYPERPARAMS[algo]
        # Variación cruzada: por cada combinación (space, start, goal) elegimos
        # un set de hiperparámetros alternando, de modo que las cuatro
        # dimensiones varían a lo largo del conjunto del algoritmo.
        combos = itertools.product(SPACE_SIZES, START_POSITIONS, GOALS)
        for idx, (space, start, goal) in enumerate(combos):
            hp = hp_options[idx % len(hp_options)]
            configs.append(
                ExperimentConfig(
                    config_id=f"{algo}-{idx:03d}",
                    algo=algo,
                    robot_name=robot,
                    space_size=space,
                    start=start,
                    goal=goal,
                    hyperparams=dict(hp),
                )
            )
    return configs


def generate_quick_configs() -> List[ExperimentConfig]:
    """Subconjunto reducido para CI: una configuración por algoritmo."""
    quick: List[ExperimentConfig] = []
    seen = set()
    for cfg in generate_configs():
        if cfg.algo in seen:
            continue
        seen.add(cfg.algo)
        quick.append(cfg)
    return quick


def _algo_class(algo: str):
    """Devuelve la clase SB3 del algoritmo (import perezoso)."""
    import stable_baselines3 as sb3  # import perezoso: solo al ejecutar

    return {"PPO": sb3.PPO, "DQN": sb3.DQN, "A2C": sb3.A2C}[algo]


def _make_backend(robot_name: str):
    """Crea el backend de ejecución.

    En Webots real usaría `from controller import Supervisor`; fuera del
    simulador (caso de esta capa de experimentos) se usa el mock del conftest
    de tests. Si no hay simulador ni mock disponible, lanza RuntimeError.
    """
    try:  # Backend mock reutilizado desde la suite de tests.
        from tests.conftest import FakeWebots  # type: ignore

        fw = FakeWebots()
        fw.install()
        return fw.make_supervisor(self_name=robot_name)
    except Exception as exc:  # pragma: no cover - defensivo
        raise RuntimeError(
            "No hay backend de Webots disponible fuera del simulador; "
            "se esperaba el mock de tests.conftest."
        ) from exc


def _train_custom(env, timesteps: int, hyperparams: Dict[str, object]):
    """Entrena el agente propio (Q-learning tabular) y devuelve el modelo.

    No usa stable-baselines3: importa el trainer del controlador custom_agent.
    `timesteps` se traduce a un número de episodios cortos (como mínimo uno)
    para mantener comparable el coste con el modo --quick de los demás.
    """
    import custom_agent  # controllers/custom_agent en el pythonpath

    # Episodios proporcionales al presupuesto de pasos; al menos 1.
    episodes = max(1, int(timesteps) // 64)
    agent = custom_agent.build_agent(env, seed=0, **hyperparams)
    custom_agent.train_agent(env, episodes=episodes, agent=agent)
    return agent


def run_config(cfg: ExperimentConfig, timesteps: int) -> Dict[str, object]:
    """Entrena brevemente y corre un episodio para `cfg`; devuelve una fila CSV.

    Los algoritmos SB3 requieren stable-baselines3 (import perezoso); el agente
    propio (CUSTOM) usa su propio trainer tabular sin SB3.
    """
    import agent_runner  # import perezoso (arrastra race_env, no SB3 al importar)

    backend = _make_backend(cfg.robot_name)
    params = race_config.RaceParams().override(
        max_dist=cfg.space_size, goal_y=cfg.goal
    )
    env = agent_runner.make_env(
        cfg.robot_name, backend, params=params, start_position=cfg.start
    )

    if cfg.algo in SB3_ALGOS:
        algo_cls = _algo_class(cfg.algo)
        model = algo_cls("MlpPolicy", env, seed=0, **cfg.hyperparams)
        model.learn(total_timesteps=timesteps)
    else:  # CUSTOM: agente propio tabular
        model = _train_custom(env, timesteps, cfg.hyperparams)

    metrics = agent_runner.run_race(model, env)
    return {
        "algo": cfg.algo,
        "config_id": cfg.config_id,
        "space_size": cfg.space_size,
        "start": "default" if cfg.start is None else str(tuple(cfg.start)),
        "goal": cfg.goal,
        "cumulative_reward": metrics["cumulative_reward"],
        "success": bool(metrics["success"]),
        "steps_to_goal": metrics["steps"] if metrics["success"] else -1,
        "falls": metrics["falls"],
        "collisions": metrics["collisions"],
    }


def run_experiments(quick: bool = False, output_csv: str = RESULTS_CSV) -> str:
    """Ejecuta el conjunto de experimentos y escribe el CSV vía pandas.

    Parameters
    ----------
    quick:
        Si True, usa el subconjunto reducido y timesteps mínimos (para CI).
    output_csv:
        Ruta del CSV de salida.

    Returns
    -------
    La ruta del CSV escrito.
    """
    import pandas as pd  # import perezoso: solo al ejecutar

    if quick:
        configs = generate_quick_configs()
        timesteps = 64  # mínimo para que SB3 cierre un rollout corto
    else:
        configs = generate_configs()
        timesteps = 20000

    rows = [run_config(cfg, timesteps) for cfg in configs]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    directory = os.path.dirname(output_csv)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Escritas {len(df)} filas en {output_csv}")
    return output_csv


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta los experimentos de la carrera DARwIn-OP."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Set reducido con timesteps mínimos (para CI).",
    )
    parser.add_argument(
        "--output",
        default=RESULTS_CSV,
        help="Ruta del CSV de salida (por defecto results/experiments.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run_experiments(quick=args.quick, output_csv=args.output)


if __name__ == "__main__":
    main()
