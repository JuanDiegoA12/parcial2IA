"""Configuración compartida de la carrera DARwIn-OP.

Define las constantes del mundo (meta, posiciones iniciales, vallas), el mapa
de acciones discretas, los umbrales de safe-learning y los pesos de recompensa.
Toda constante es sobre-escribible vía `RaceParams` para poder variar el
tamaño del espacio, la posición inicial y la meta en los experimentos.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

# --- Geometría del mundo (eje de carrera = y) -------------------------------

# Plano de la meta (DEF START_FINISH en y=9). Cruzar GOAL_Y gana la carrera.
GOAL_Y: float = 9.0

# Posiciones iniciales exactas tomadas del .wbt, por nombre exacto del robot.
# (x, y, z) en metros. El nombre 'actor - critic' lleva los espacios literales.
START_POSITIONS: Dict[str, Tuple[float, float, float]] = {
    "DQN": (0.11, 9.95, 0.24),
    "ppo": (0.21, 7.88, 0.34),
    "actor - critic": (0.21, 6.04, 0.34),
    "agente propio": (0.21, 11.99, 0.34),
}

# Mapa de acciones discretas -> método de locomoción en DarwinRobot.
ACTION_MAP: Dict[int, str] = {
    0: "walk_forward",
    1: "turn_left",
    2: "turn_right",
}

# --- Safe learning ----------------------------------------------------------

# Radio de seguridad (m): acercarse a un obstáculo u otro robot dentro de este
# radio genera penalización proporcional a la intrusión.
SAFETY_RADIUS: float = 0.6

# Peso de la penalización por intrusión en el radio de seguridad de obstáculos.
OBSTACLE_PENALTY_WEIGHT: float = 2.0
# Peso de la penalización por intrusión en el radio de seguridad de otros robots.
OTHER_ROBOT_PENALTY_WEIGHT: float = 2.0
# Distancia (m) considerada colisión dura -> termina el episodio.
HARD_COLLISION_DISTANCE: float = 0.15

# Umbrales de caída: inclinación del torso (rad) a partir de la cual el robot
# se considera caído. Se evalúa con acelerómetro/giroscopio.
FALL_TILT_THRESHOLD: float = 0.6  # ~34 grados

# --- Episodio ---------------------------------------------------------------

MAX_EPISODE_STEPS: int = 400

# --- Normalización de la observación ----------------------------------------

# Cotas usadas para normalizar cada componente de la observación a [-1, 1] o
# [0, 1]. MAX_DIST acota distancias; el ángulo ya está en [-pi, pi].
MAX_DIST: float = 60.0  # diagonal aproximada del espacio de carrera (m)

# --- Pesos de recompensa ----------------------------------------------------

# Recompensa por metro de avance hacia la meta (reducción de dist_to_goal).
PROGRESS_WEIGHT: float = 10.0
# Bono grande al cruzar la meta.
GOAL_BONUS: float = 100.0
# Penalización al caer.
FALL_PENALTY: float = 50.0
# Penalización por colisión dura.
COLLISION_PENALTY: float = 50.0
# Pequeño costo por paso para incentivar terminar rápido.
STEP_PENALTY: float = 0.1


@dataclass
class RaceParams:
    """Parámetros sobre-escribibles de un experimento de carrera.

    Permite variar tamaño del espacio, posición inicial y meta sin tocar el
    código del entorno. Los valores por defecto replican las constantes de
    módulo (el mundo .wbt real).
    """

    goal_y: float = GOAL_Y
    start_positions: Dict[str, Tuple[float, float, float]] = field(
        default_factory=lambda: dict(START_POSITIONS)
    )
    safety_radius: float = SAFETY_RADIUS
    obstacle_penalty_weight: float = OBSTACLE_PENALTY_WEIGHT
    other_robot_penalty_weight: float = OTHER_ROBOT_PENALTY_WEIGHT
    hard_collision_distance: float = HARD_COLLISION_DISTANCE
    fall_tilt_threshold: float = FALL_TILT_THRESHOLD
    max_episode_steps: int = MAX_EPISODE_STEPS
    max_dist: float = MAX_DIST
    progress_weight: float = PROGRESS_WEIGHT
    goal_bonus: float = GOAL_BONUS
    fall_penalty: float = FALL_PENALTY
    collision_penalty: float = COLLISION_PENALTY
    step_penalty: float = STEP_PENALTY

    def override(self, **kwargs) -> "RaceParams":
        """Devuelve una copia con los campos indicados reemplazados."""
        return replace(self, **kwargs)


# Instancia por defecto reutilizable.
DEFAULT_PARAMS = RaceParams()
