"""Configuración compartida de la carrera DARwIn-OP.

Define las constantes del mundo (meta, posiciones iniciales, vallas), el mapa
de acciones discretas, los umbrales de safe-learning y los pesos de recompensa.
Toda constante es sobre-escribible vía `RaceParams` para poder variar el
tamaño del espacio, la posición inicial y la meta en los experimentos.
"""

from __future__ import annotations

import math
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
# se considera caído. Se evalúa con la orientación del Supervisor y/o la
# desviación del acelerómetro. AJUSTABLE en Webots.
FALL_TILT_THRESHOLD: float = 0.6  # ~34 grados

# Modelo del acelerómetro del DARwIn-OP en Webots: de pie cada eje lee
# centrado en ~512 (NO un modelo (0,0,9.81)). Walk.cpp compara acc[1] contra
# 512 +/- 80. Caída = desviación de ACCEL_CENTER mayor que ACCEL_FALL_TOLERANCE.
ACCEL_CENTER: float = 512.0           # lectura nominal por eje de pie. AJUSTABLE
ACCEL_FALL_TOLERANCE: float = 80.0    # banda tolerada antes de declarar caída. AJUSTABLE

# Altura (z, m) del torso por debajo de la cual el robot se considera caído.
# De pie el DARwIn-OP ronda z~0.24-0.34 según el .wbt; si colapsa cae bajo esto.
FALL_HEIGHT_Z: float = 0.18           # AJUSTABLE en Webots segun la pose real

# --- Episodio ---------------------------------------------------------------

MAX_EPISODE_STEPS: int = 400

# --- Locomoción: gait sinusoidal (DARwIn-OP) --------------------------------
# Toda la locomoción vive en darwin_robot.py (_GaitEngine); aquí están TODAS
# las constantes ajustables. El robot NO usa los managers de Robotis (son C++);
# accionamos los 20 motores directamente con setPosition cada paso.

# Los 20 motores del DARwIn-OP en el orden del controlador oficial Walk.cpp.
# Cada motor tiene un PositionSensor llamado <nombre>+"S".
MOTOR_NAMES = (
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR", "ArmLowerL",
    "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR", "LegUpperL",
    "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR", "FootL",
    "Neck", "Head",
)

# Pose nominal estable de pie (motor -> ángulo en rad). Brazos ligeramente
# separados, piernas flexionadas en cadera/rodilla/tobillo para bajar el centro
# de masa. Es el punto de reposo sobre el que oscila el gait. AJUSTABLE.
STANDING_POSE: Dict[str, float] = {
    "ShoulderR": 0.8, "ShoulderL": -0.8,
    "ArmUpperR": 0.5, "ArmUpperL": -0.5,
    "ArmLowerR": -0.6, "ArmLowerL": 0.6,
    "PelvYR": 0.0, "PelvYL": 0.0,
    "PelvR": 0.0, "PelvL": 0.0,
    "LegUpperR": 0.5, "LegUpperL": -0.5,
    "LegLowerR": 1.0, "LegLowerL": -1.0,
    "AnkleR": -0.5, "AnkleL": 0.5,
    "FootR": 0.0, "FootL": 0.0,
    "Neck": 0.0, "Head": 0.0,
}

# Periodo del gait en pasos de simulación (un ciclo completo de ambas piernas).
# La frecuencia angular por tick es 2*pi / GAIT_PERIOD. AJUSTABLE: subirlo da
# pasos más lentos y estables; bajarlo, más rápidos.
GAIT_PERIOD: int = 40
# Frecuencia derivada (ciclos por paso de simulación), documentada para claridad.
GAIT_FREQUENCY: float = 1.0 / GAIT_PERIOD

# Amplitudes de "comando" análogas a setXAmplitude/setAAmplitude del C++.
# El agente elige una acción discreta y el motor de gait la traduce a estas
# amplitudes. Valores en [0, 1] que escalan las amplitudes articulares. AJUSTABLE.
GAIT_X_AMPLITUDE: float = 1.0   # avance (walk_forward)
GAIT_A_AMPLITUDE: float = 0.6   # giro (turn_left/right)

# Amplitudes articulares base (rad) del gait, escaladas por x/a. AJUSTABLE.
GAIT_HIP_AMPLITUDE: float = 0.35    # oscilación de cadera (LegUpper)
GAIT_KNEE_AMPLITUDE: float = 0.4    # flexión de rodilla (LegLower)
GAIT_ANKLE_AMPLITUDE: float = 0.2   # tobillo (Ankle)
GAIT_LATERAL_SWAY: float = 0.12     # balanceo lateral de peso (Pelv/Ankle). AJUSTABLE

# Desfase (rad) entre piernas: izquierda y derecha a 180 grados. AJUSTABLE.
GAIT_LEG_PHASE_OFFSET: float = math.pi

# Velocidad por defecto de los motores (rad/s) al accionarlos. AJUSTABLE.
MOTOR_VELOCITY: float = 8.0

# Límites articulares (motor -> (min, max) en rad). La rodilla LegLowerR/L es
# fija en [-2.25, 2.25] (dato del enunciado/modelo). El resto son rangos
# conservadores y seguros: NO se consultó el PROTO Darwin-op.proto en el
# sandbox (sin red), así que se asumen amplios pero prudentes. Ajustar si se
# observa recorte excesivo contra los límites reales del PROTO. ASUNCIÓN.
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "ShoulderR": (-3.14, 3.14), "ShoulderL": (-3.14, 3.14),
    "ArmUpperR": (-3.14, 3.14), "ArmUpperL": (-3.14, 3.14),
    "ArmLowerR": (-1.57, 1.57), "ArmLowerL": (-1.57, 1.57),
    "PelvYR": (-1.57, 1.57), "PelvYL": (-1.57, 1.57),
    "PelvR": (-1.57, 1.57), "PelvL": (-1.57, 1.57),
    "LegUpperR": (-1.57, 1.57), "LegUpperL": (-1.57, 1.57),
    "LegLowerR": (-2.25, 2.25), "LegLowerL": (-2.25, 2.25),
    "AnkleR": (-1.57, 1.57), "AnkleL": (-1.57, 1.57),
    "FootR": (-1.57, 1.57), "FootL": (-1.57, 1.57),
    "Neck": (-1.57, 1.57), "Head": (-0.96, 0.96),
}

# Margen de seguridad (rad) que NUNCA se consume: todo ángulo se recorta a
# [min+margen, max-margen]. Evita las advertencias "too big/low requested
# position" de Webots. AJUSTABLE.
JOINT_LIMIT_MARGIN: float = 0.05

# Pasos de estabilización: ticks quietos manteniendo la pose de pie antes de
# empezar cada episodio, para que la física asiente al robot. AJUSTABLE.
STABILIZATION_STEPS: int = 20

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
