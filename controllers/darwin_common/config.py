"""Configuración compartida de la carrera DARwIn-OP.

Define las constantes del mundo (meta, posiciones iniciales, vallas), el mapa
de acciones discretas, los umbrales de safe-learning y los pesos de recompensa.
Toda constante es sobre-escribible vía `RaceParams` para poder variar el
tamaño del espacio, la posición inicial y la meta en los experimentos.
"""

from __future__ import annotations

import math
import os
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
# De pie el DARwIn-OP ronda z~0.23-0.25 (nominal confirmado z~0.247); colapsado
# cae a ~0.075. FALL_HEIGHT_Z va claramente por debajo de la pose de pie y por
# encima del colapso, de modo que de pie/en cuclillas NO cuenta como caída pero
# el colapso sí. Es la señal PRIMARIA de caída. AJUSTABLE en Webots.
FALL_HEIGHT_Z: float = 0.14           # AJUSTABLE en Webots segun la pose real

# Persistencia del acelerómetro antes de declarar caída, al estilo del
# controlador oficial Walk.cpp: una lectura puntual fuera de banda NO basta;
# hacen falta FALL_ACCEL_PERSIST_STEPS ticks consecutivos fuera de banda. Evita
# que un pico de ruido del acelerómetro marque fallen=True. AJUSTABLE.
FALL_ACCEL_PERSIST_STEPS: int = 100

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
# separados; piernas en una flexión MODESTA de cadera/rodilla/tobillo (cuclillas
# suave) para bajar el centro de masa y estabilizar — el usuario confirmó que el
# robot sólo se sostiene flexionado, nunca totalmente erguido. Es el punto de
# reposo sobre el que oscila el gait.
#
# IMPORTANTE (modelo Darwin-op de Webots): las articulaciones de pierna/brazo
# están ESPEJADAS con rangos de signo invertido entre izquierda y derecha (ver
# JOINT_LIMITS). Por eso la flexión de la rodilla derecha es POSITIVA
# (LegLowerR en [-0.02, 2.25]) y la izquierda NEGATIVA (LegLowerL en
# [-2.25, 0.03]); la cadera derecha es NEGATIVA (LegUpperR en [-1.77, 0.45]) y
# la izquierda POSITIVA (LegUpperL en [-0.5, 1.68]).
#
# TODOS estos valores son PUNTOS DE PARTIDA para AJUSTAR en Webots: ejecuta
# locomotion_test con LOCO_CMD=stand, observa la z resuelta (~0.23-0.25) y el
# equilibrio, y retoca cadera/rodilla/tobillo. Cada valor queda al menos
# JOINT_LIMIT_MARGIN dentro de su rango REAL para que el recorte nunca dispare.
STANDING_POSE: Dict[str, float] = {
    "ShoulderR": 0.8, "ShoulderL": -0.8,
    "ArmUpperR": 0.4, "ArmUpperL": -0.4,
    "ArmLowerR": -0.6, "ArmLowerL": 0.6,
    "PelvYR": 0.0, "PelvYL": 0.0,
    "PelvR": 0.0, "PelvL": 0.0,
    "LegUpperR": -0.6, "LegUpperL": 0.6,    # cadera: R neg, L pos (espejo)
    "LegLowerR": 1.2, "LegLowerL": -1.2,    # rodilla: R pos, L neg (espejo)
    "AnkleR": -0.6, "AnkleL": 0.6,          # tobillo: R neg, L pos (espejo)
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

# Límites articulares (motor -> (min, max) en rad). Son un FALLBACK: la fuente
# autoritativa son los límites REALES que _LocomotionBackend lee en runtime con
# motor.getMinPosition()/getMaxPosition() (ver darwin_robot.py) y que
# locomotion_test vuelca por consola en su modo diagnóstico. Este fallback se
# usa cuando esos getters no están disponibles (p.ej. el mock en tests).
#
# Valores tomados del PROTO oficial Darwin-op de Webots R2025a
# (projects/robots/robotis/darwin-op/protos/Darwin-op.proto). Son ASIMÉTRICOS y
# están ESPEJADOS con el signo invertido entre izquierda y derecha: NO uses el
# placeholder simétrico (-1.57, 1.57) de antes. En particular LegUpperR tope
# 0.45 explica la advertencia "LegUpperR 0.5 > 0.45". El PROTO es remoto, así
# que offline no puede reparsearse; si Webots imprime otros topes, confía en los
# de runtime. (Contenido reformulado por cumplimiento de licencia.)
JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "ShoulderR": (-3.14, 3.14), "ShoulderL": (-3.14, 3.14),
    "ArmUpperR": (-0.68, 2.30), "ArmUpperL": (-2.25, 0.77),
    "ArmLowerR": (-1.65, 1.16), "ArmLowerL": (-1.18, 1.63),
    "PelvYR": (-2.42, 0.66), "PelvYL": (-0.69, 2.50),
    "PelvR": (-1.01, 1.01), "PelvL": (-1.00, 0.93),
    "LegUpperR": (-1.77, 0.45), "LegUpperL": (-0.50, 1.68),
    "LegLowerR": (-0.02, 2.25), "LegLowerL": (-2.25, 0.03),
    "AnkleR": (-1.24, 1.38), "AnkleL": (-1.39, 1.22),
    "FootR": (-0.68, 1.04), "FootL": (-1.02, 0.60),
    "Neck": (-1.81, 1.81), "Head": (-0.36, 0.94),
}

# Margen de seguridad (rad) que NUNCA se consume: todo ángulo se recorta a
# [min+margen, max-margen]. Evita las advertencias "too big/low requested
# position" de Webots. AJUSTABLE.
JOINT_LIMIT_MARGIN: float = 0.05

# Pasos de estabilización: ticks quietos manteniendo la pose de pie antes de
# empezar cada episodio, para que la física asiente al robot en la pose de
# cuclillas. Debe ser lo bastante largo para que la pose se asiente antes de
# iniciar el comando/episodio; súbelo si el robot aún no está quieto al empezar
# a moverse. AJUSTABLE en Webots.
STABILIZATION_STEPS: int = 80

# --- Modo de locomoción: Plan A (gait) vs Plan B (cinemático) ---------------
#
# El proyecto se centra en la CARRERA y en evaluar agentes de RL (PPO, DQN,
# A2C y agente propio), NO en caminar. Por eso existen DOS modos de locomoción
# intercambiables, seleccionables con una sola bandera:
#
#   Plan A -- "gait": gait sinusoidal físico que acciona los 20 motores del
#       DARwIn-OP cada paso (_GaitEngine en darwin_robot.py). Es la locomoción
#       "de verdad", pero costosa de ajustar y sensible al equilibrio.
#
#   Plan B -- "kinematic": arrastre cinemático (tipo "tortuga" diferencial).
#       El Supervisor desliza al robot DE PIE escribiendo directamente los
#       campos translation/rotation cada paso (teletransporte por paso), sin
#       depender del equilibrio de los motores. Es una DECISIÓN DE DISEÑO
#       deliberada y aceptada: para el objetivo (carrera + evaluación de
#       agentes) el movimiento es un lujo, no una necesidad, y este modo es
#       "time-safe" (no hay que ajustar la marcha física). Mantiene las MISMAS
#       acciones discretas, el MISMO DarwinRaceEnv y los MISMOS agentes.
#
# En modo kinematic la caída se desactiva (is_fallen() devuelve False), pero
# las penalizaciones por proximidad/colisión y el cruce de GOAL_Y siguen
# contando exactamente igual que en modo gait.
#
# QUÉ VERIFICAR EN WEBOTS REAL (el modo kinematic refinado; aquí no hay runtime):
#   1) velocidad ~1 m/s con LOCO_CMD=forward (KINEMATIC_LINEAR_STEP a 16 ms/tick);
#   2) el robot se DETIENE al llegar a una valla (no la penetra) y puede girar
#      para esquivarla (el giro nunca se bloquea);
#   3) el torso se mantiene a z ~ KINEMATIC_STAND_Z (~0.24), sin acostarse,
#      y la orientación es yaw puro (sin roll/pitch);
#   4) is_fallen() sigue devolviendo False en kinematic.

# Modo por defecto: 'kinematic' (Plan B), por DECISIÓN EXPLÍCITA del usuario.
# El foco del trabajo es la carrera y la evaluación de agentes (PPO/DQN/A2C/
# propio), no la marcha humanoide; el modo 'kinematic' arrastra al robot de pie
# vía Supervisor (translation/rotation), is_fallen() queda en False, pero las
# penalizaciones por proximidad/colisión y el cruce de GOAL_Y SIGUEN activos.
# El modo 'gait' (Plan A, gait sinusoidal por motores con detección de caída)
# queda disponible como opción, seleccionable con la variable de entorno
# LOCOMOTION_MODE=gait o cambiando esta constante. Valores válidos: 'gait' |
# 'kinematic'.
LOCOMOTION_MODE: str = "kinematic"

# Paso lineal del arrastre cinemático: metros que avanza el robot por cada tick
# de walk_forward, a lo largo de su heading. ESTA es la ÚNICA fuente de la
# velocidad: tanto el desplazamiento real en Webots (darwin_robot.py) como el
# mock de los tests (tests/conftest.py, WALK_DELTA) leen esta constante.
# Derivación a basicTimeStep = 16 ms = 0.016 s/tick:
#   0.016 m/tick / 0.016 s/tick = 1.0 m/s  (paso humano, realista para la carrera).
# Antes era 0.25 m/tick = ~15.6 m/s (irreal; hacía trivial la carrera). AJUSTABLE.
KINEMATIC_LINEAR_STEP: float = 0.016

# Paso angular del arrastre cinemático: radianes que rota el robot por cada
# tick de turn_left/turn_right. Igual al TURN_DELTA del mock (misma fuente única).
# Derivación a 16 ms/tick: 0.04 rad/tick / 0.016 s/tick = 2.5 rad/s ≈ 143 °/s
#   (giro suave pero usable). Antes 0.20 rad/tick = 12.5 rad/s ≈ 716 °/s. AJUSTABLE.
KINEMATIC_ANGULAR_STEP: float = 0.04

# Radio de colisión del arrastre cinemático (m): antes de COMPROMETER un avance
# de walk_forward se calcula la (x,y) de destino; si cae dentro de este radio de
# alguna valla (find_barriers) u otro robot (find_others) el avance se CANCELA
# para ese tick (el robot no penetra el obstáculo). El giro NUNCA se bloquea, de
# modo que el agente puede reorientarse y esquivar. Valor ~ semiancho del robot
# (~0.15) + semiespesor de la valla (~0.15). Queda por DEBAJO de SAFETY_RADIUS
# (0.6), para que la penalización por proximidad siga disparándose ANTES del
# bloqueo, y por ENCIMA de HARD_COLLISION_DISTANCE (0.15). AJUSTABLE en Webots.
KINEMATIC_COLLISION_RADIUS: float = 0.3

# Altura (z, m) a la que se fija el torso en el arrastre cinemático: pin visual
# de pie. En vez de heredar la z colapsada, cada escritura de translation en
# modo kinematic fija z = KINEMATIC_STAND_Z para que el robot se deslice DE PIE.
# Coincide con la z nominal de pie del DARwIn-OP (~0.24, rango ~0.23–0.25) y con
# la z de arranque de DQN. Es una APROXIMACIÓN visual/cinemática: el usuario debe
# verificar en Webots que el torso no se acuesta (ver notas arriba). AJUSTABLE.
KINEMATIC_STAND_Z: float = 0.24


def resolve_locomotion_mode() -> str:
    """Devuelve el modo de locomoción efectivo ('gait' o 'kinematic').

    Prioridad: variable de entorno LOCOMOTION_MODE (si está definida y es
    válida) sobre la constante de módulo LOCOMOTION_MODE. El valor se
    normaliza a minúsculas y se valida contra {'gait', 'kinematic'}; cualquier
    valor no reconocido cae al valor por defecto LOCOMOTION_MODE.
    """
    valid = {"gait", "kinematic"}
    raw = os.environ.get("LOCOMOTION_MODE", LOCOMOTION_MODE)
    mode = str(raw).strip().lower()
    if mode not in valid:
        return LOCOMOTION_MODE
    return mode


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
