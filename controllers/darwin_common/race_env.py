"""Entorno gymnasium de la carrera DARwIn-OP, consumido por SB3.

La pista es OVALADA y el inicio coincide con el fin (los agentes arrancan en
y~9, que es también la meta y=GOAL_Y). Para que ganar exija dar la VUELTA, la
recompensa y la condición de victoria son por CHECKPOINTS ORDENADOS: el robot
debe alcanzar cp1..cp7 en orden y luego volver a cp0 (la meta). Cruzar y=9 sin
completar la secuencia NO gana.

Observación: Box de 7 floats normalizados
    [dist_to_next_cp, angle_to_next_cp, heading, lap_progress,
     dist_to_finish, dist_nearest_obstacle, dist_nearest_other_robot]
donde dist_to_finish = distancia en línea recta a cp0 (cierre de vuelta)
normalizada por max_dist; es una señal continua distinta de lap_progress
(escalón discreto reached_count/total).
Acción: Discrete(3) -> config.ACTION_MAP (0 avanzar, 1 girar izq, 2 girar der).

Recompensa (safe learning):
    + progreso continuo hacia el SIGUIENTE checkpoint (reducción de distancia)
    + bono discreto al ALCANZAR cada checkpoint; GOAL_BONUS y `terminated` al
      cerrar la vuelta (volver a cp0 tras cp1..cp7 en orden)
    - penalización por salirse de la línea de carrera (distancia al segmento
      predecesor->objetivo más allá de OFF_TRACK_TOLERANCE)
    - penalización proporcional a la intrusión en SAFETY_RADIUS de obstáculos
      y otros robots
    - penalización y `terminated` al caer o colisionar duro
    - pequeño costo por paso
    truncated al alcanzar MAX_EPISODE_STEPS.

El backend (Webots real o mock) se inyecta: el entorno no importa Webots.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from . import config
    from .darwin_robot import DarwinRobot
except ImportError:  # pragma: no cover - ruta bajo pythonpath plano
    import config
    from darwin_robot import DarwinRobot


def _segment_distance(px: float, py: float,
                      ax: float, ay: float,
                      bx: float, by: float) -> float:
    """Distancia del punto P=(px,py) al segmento A=(ax,ay)->B=(bx,by).

    Proyecta P sobre el segmento con el parámetro t recortado a [0, 1]. Si el
    segmento es degenerado (A == B) devuelve la distancia punto-punto a A.
    Determinista; sin dependencias más allá de math.
    """
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


class CheckpointTracker:
    """Máquina de estados de la vuelta de checkpoints (Python puro).

    Compartida por DarwinRaceEnv y RaceWinnerDetector para que la recompensa y
    la condición de victoria coincidan por construcción. No depende de Webots ni
    de numpy.

    Parameters
    ----------
    targets:
        Lista ORDENADA de coordenadas (x, y) de los objetivos de la vuelta, en
        el orden de config.CHECKPOINT_ORDER: [cp1, cp2, ..., cp7, cp0].
    radius:
        Radio (m) para considerar alcanzado un objetivo (CHECKPOINT_RADIUS).
    origin:
        Punto de partida (x, y) usado como predecesor del PRIMER objetivo para
        el segmento de la línea de carrera. Por defecto cp0 = (0, GOAL_Y) (el
        último elemento de `targets` en el orden canónico).
    """

    def __init__(self, targets: Sequence[Tuple[float, float]],
                 radius: float = config.CHECKPOINT_RADIUS,
                 origin: Optional[Tuple[float, float]] = None):
        self.targets: List[Tuple[float, float]] = [
            (float(x), float(y)) for (x, y) in targets
        ]
        self.radius = float(radius)
        # Predecesor del primer objetivo: el origen (cp0) si no se indica otro.
        if origin is not None:
            self.origin = (float(origin[0]), float(origin[1]))
        elif self.targets:
            # cp0 es el ÚLTIMO objetivo en el orden canónico (cierre de vuelta).
            self.origin = self.targets[-1]
        else:
            self.origin = (0.0, 0.0)
        self.reset()

    def reset(self) -> None:
        self.reached_count = 0
        self.prev_dist: Optional[float] = None
        self.done = False

    def current_target(self) -> Optional[Tuple[float, float]]:
        """Coordenadas del objetivo actual, o None si la vuelta está completa."""
        if self.done or self.reached_count >= len(self.targets):
            return None
        return self.targets[self.reached_count]

    def _predecessor(self) -> Tuple[float, float]:
        """Punto predecesor del objetivo actual (origen o checkpoint previo)."""
        if self.reached_count <= 0:
            return self.origin
        return self.targets[self.reached_count - 1]

    def current_segment(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """(predecesor, objetivo) del tramo actual para la penalización off-track."""
        target = self.current_target()
        if target is None:
            return None
        return (self._predecessor(), target)

    def lap_progress(self) -> float:
        """Fracción de la vuelta completada: reached_count / total objetivos."""
        if not self.targets:
            return 1.0
        return self.reached_count / len(self.targets)

    def finish_point(self) -> Tuple[float, float]:
        """Coordenadas del cierre de vuelta (cp0), último objetivo del orden."""
        if self.targets:
            return self.targets[-1]
        return self.origin

    def update(self, px: float, py: float) -> Dict[str, object]:
        """Procesa una posición (px, py); devuelve el resultado del tick.

        Calcula el progreso (reducción de distancia al objetivo actual desde la
        llamada previa), avanza el objetivo si se alcanzó (dentro de radius) y
        marca done al cerrar la vuelta (alcanzar el último objetivo = cp0).

        Returns
        -------
        dict con: progress (float), reached (bool), done (bool),
        reached_count (int), target_ordinal (int), off_segment (tupla o None).
        """
        if self.done:
            return {
                "progress": 0.0,
                "reached": False,
                "done": True,
                "reached_count": self.reached_count,
                "target_ordinal": self.reached_count,
                "off_segment": None,
            }

        target = self.current_target()
        segment = self.current_segment()
        dist = math.hypot(px - target[0], py - target[1])
        progress = (self.prev_dist - dist) if self.prev_dist is not None else 0.0

        reached = dist <= self.radius
        if reached:
            self.reached_count += 1
            # Tras avanzar, reiniciar prev_dist contra el nuevo objetivo (o None
            # si la vuelta terminó), para que el próximo progreso parta de cero.
            if self.reached_count >= len(self.targets):
                self.done = True
                self.prev_dist = None
            else:
                nxt = self.targets[self.reached_count]
                self.prev_dist = math.hypot(px - nxt[0], py - nxt[1])
        else:
            self.prev_dist = dist

        return {
            "progress": float(progress),
            "reached": bool(reached),
            "done": bool(self.done),
            "reached_count": int(self.reached_count),
            "target_ordinal": int(self.reached_count),
            "off_segment": segment,
        }


class DarwinRaceEnv(gym.Env):
    """Entorno de carrera para un robot DARwIn-OP contra sus rivales."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend,
        robot_name: str = "DQN",
        params: Optional[config.RaceParams] = None,
        start_position: Optional[Tuple[float, float, float]] = None,
        goal_y: Optional[float] = None,
    ):
        super().__init__()
        self.params = params or config.RaceParams()
        if goal_y is not None:
            self.params = self.params.override(goal_y=goal_y)
        self.robot_name = robot_name
        self.robot = DarwinRobot(backend, robot_name)

        # Posición inicial: override explícito o la de config por nombre.
        if start_position is not None:
            self.start_position = tuple(start_position)
        else:
            self.start_position = self.params.start_positions.get(
                robot_name, (0.0, 0.0, 0.3)
            )

        # Secuencia ORDENADA de checkpoints leída del mundo (cp0 sintetizado),
        # fuente única compartida con el detector de ganador (race.py).
        targets = self.robot.ordered_checkpoints(
            order=self.params.checkpoint_order, goal_y=self.params.goal_y
        )
        self.tracker = CheckpointTracker(
            targets, radius=self.params.checkpoint_radius
        )

        self.action_space = spaces.Discrete(len(config.ACTION_MAP))
        # dist_next_cp, angle_next_cp, heading, lap_progress, dist_to_finish,
        # dist_obstacle, dist_other
        high = np.ones(7, dtype=np.float32)
        low = np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._steps = 0

    # --- API gymnasium ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        x, y, z = self.start_position
        self.robot.reset_pose(x, y, z)
        self._steps = 0
        self.tracker.reset()
        # Inicializa prev_dist del tracker contra la pose de arranque para que
        # el primer step mida progreso real (sin un salto espurio).
        px, py, _ = self.robot.get_position()
        target = self.tracker.current_target()
        if target is not None:
            self.tracker.prev_dist = math.hypot(px - target[0], py - target[1])
        return self._get_obs(), self._base_info()

    def step(self, action):
        action = int(action)
        method = config.ACTION_MAP[action]
        getattr(self.robot, method)()
        self.robot.tick()
        self._steps += 1

        px, py, _ = self.robot.get_position()
        result = self.tracker.update(px, py)
        progress = float(result["progress"])

        nearest_obstacle = self._nearest_distance(self.robot.find_barriers())
        nearest_other = self._nearest_distance(self.robot.find_others())

        reward = (
            self.params.checkpoint_progress_weight * progress
            - self.params.step_penalty
        )
        terminated = False
        truncated = False

        # Bono discreto al alcanzar un checkpoint intermedio.
        if result["reached"] and not result["done"]:
            reward += self.params.checkpoint_bonus

        # Penalización por salirse de la línea de carrera (segmento actual).
        off_segment = result["off_segment"]
        if off_segment is not None:
            (ax, ay), (bx, by) = off_segment
            deviation = _segment_distance(px, py, ax, ay, bx, by)
            if deviation > self.params.off_track_tolerance:
                reward -= self.params.off_track_penalty_weight * (
                    deviation - self.params.off_track_tolerance
                )

        fall = self.robot.is_fallen()
        collision = False
        margin_violations = 0

        # Intrusión en radio de seguridad (obstáculos).
        if nearest_obstacle is not None and nearest_obstacle < self.params.safety_radius:
            intrusion = self.params.safety_radius - nearest_obstacle
            reward -= self.params.obstacle_penalty_weight * intrusion
            margin_violations += 1
            if nearest_obstacle < self.params.hard_collision_distance:
                collision = True
        # Intrusión en radio de seguridad (otros robots).
        if nearest_other is not None and nearest_other < self.params.safety_radius:
            intrusion = self.params.safety_radius - nearest_other
            reward -= self.params.other_robot_penalty_weight * intrusion
            margin_violations += 1
            if nearest_other < self.params.hard_collision_distance:
                collision = True

        # Éxito = vuelta COMPLETA (cp1..cp7 en orden y retorno a cp0). Cruzar
        # y=9 sin la secuencia NO gana.
        success = bool(result["done"])
        if success:
            reward += self.params.goal_bonus
            terminated = True

        if fall:
            reward -= self.params.fall_penalty
            terminated = True
        if collision:
            reward -= self.params.collision_penalty
            terminated = True

        if self._steps >= self.params.max_episode_steps:
            truncated = True

        info = self._base_info()
        info.update(
            {
                "fall": fall,
                "collision": collision,
                "margin_violations": margin_violations,
                "progress": float(progress),
                "success": success,
                "checkpoints_reached": int(result["reached_count"]),
            }
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    # --- observación y helpers ---------------------------------------------
    def _target_point(self) -> Tuple[float, float]:
        """Objetivo actual, o el último objetivo (cp0) si la vuelta terminó.

        Mantiene la observación finita y en rango tras completar la vuelta.
        """
        target = self.tracker.current_target()
        if target is not None:
            return target
        if self.tracker.targets:
            return self.tracker.targets[-1]
        return (0.0, self.params.goal_y)

    def _dist_to_target(self) -> float:
        px, py, _ = self.robot.get_position()
        tx, ty = self._target_point()
        return math.hypot(px - tx, py - ty)

    def _dist_to_finish(self) -> float:
        """Distancia en línea recta del robot al cierre de vuelta (cp0).

        Señal continua de "cuánto falta para cerrar la vuelta", distinta del
        lap_progress (escalón discreto reached_count/total). Referida siempre a
        cp0 = (0, GOAL_Y) sin importar el objetivo actual.
        """
        px, py, _ = self.robot.get_position()
        fx, fy = self.tracker.finish_point()
        return math.hypot(px - fx, py - fy)

    def _angle_to_target(self) -> float:
        px, py, _ = self.robot.get_position()
        tx, ty = self._target_point()
        dx = tx - px
        dy = ty - py
        # Convención de heading: 0 mira a +y; avance = (-sin, cos). El ángulo al
        # objetivo usa atan2(-dx, dy) para ser consistente con get_orientation.
        target_heading = math.atan2(-dx, dy) if (dx or dy) else 0.0
        heading = self.robot.get_orientation()
        diff = target_heading - heading
        return math.atan2(math.sin(diff), math.cos(diff))

    def _nearest_distance(self, nodes: Dict[str, Tuple[float, float]]) -> Optional[float]:
        if not nodes:
            return None
        px, py, _ = self.robot.get_position()
        return min(math.hypot(px - x, py - y) for (x, y) in nodes.values())

    def _get_obs(self) -> np.ndarray:
        dist = self._dist_to_target()
        angle = self._angle_to_target()
        heading = self.robot.get_orientation()
        nearest_obstacle = self._nearest_distance(self.robot.find_barriers())
        nearest_other = self._nearest_distance(self.robot.find_others())

        max_dist = self.params.max_dist
        obs = np.array(
            [
                min(dist / max_dist, 1.0),
                max(-1.0, min(angle / math.pi, 1.0)),
                max(-1.0, min(heading / math.pi, 1.0)),
                max(0.0, min(self.tracker.lap_progress(), 1.0)),
                max(0.0, min(self._dist_to_finish() / max_dist, 1.0)),
                1.0 if nearest_obstacle is None else min(nearest_obstacle / max_dist, 1.0),
                1.0 if nearest_other is None else min(nearest_other / max_dist, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _base_info(self) -> dict:
        px, py, _ = self.robot.get_position()
        return {"robot_name": self.robot_name, "x": float(px), "y": float(py), "steps": self._steps}
