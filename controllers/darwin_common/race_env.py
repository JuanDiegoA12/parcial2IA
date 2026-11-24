"""Entorno gymnasium de la carrera DARwIn-OP, consumido por SB3.

Observación: Box de 5 floats normalizados
    [dist_to_goal, angle_to_goal, heading, dist_nearest_obstacle,
     dist_nearest_other_robot]
Acción: Discrete(3) -> config.ACTION_MAP (0 avanzar, 1 girar izq, 2 girar der).

Recompensa (safe learning):
    + progreso hacia la meta (reducción de dist_to_goal)
    + bono grande y `terminated` al cruzar GOAL_Y
    - penalización proporcional a la intrusión en SAFETY_RADIUS de obstáculos
      y otros robots
    - penalización y `terminated` al caer o colisionar duro
    truncated al alcanzar MAX_EPISODE_STEPS.

El backend (Webots real o mock) se inyecta: el entorno no importa Webots.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from . import config
    from .darwin_robot import DarwinRobot
except ImportError:  # pragma: no cover - ruta bajo pythonpath plano
    import config
    from darwin_robot import DarwinRobot


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

        self.action_space = spaces.Discrete(len(config.ACTION_MAP))
        # dist_to_goal, angle_to_goal, heading, dist_obstacle, dist_other
        high = np.ones(5, dtype=np.float32)
        low = np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._steps = 0
        self._prev_dist = None

    # --- API gymnasium ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        x, y, z = self.start_position
        self.robot.reset_pose(x, y, z)
        self._steps = 0
        self._prev_dist = self._dist_to_goal()
        return self._get_obs(), self._base_info()

    def step(self, action):
        action = int(action)
        method = config.ACTION_MAP[action]
        getattr(self.robot, method)()
        self.robot.tick()
        self._steps += 1

        dist = self._dist_to_goal()
        progress = (self._prev_dist - dist) if self._prev_dist is not None else 0.0
        self._prev_dist = dist

        px, py, _ = self.robot.get_position()
        nearest_obstacle = self._nearest_distance(self.robot.find_barriers())
        nearest_other = self._nearest_distance(self.robot.find_others())

        reward = self.params.progress_weight * progress - self.params.step_penalty
        terminated = False
        truncated = False

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

        success = py >= self.params.goal_y
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
                "success": bool(success),
            }
        )
        return self._get_obs(), float(reward), terminated, truncated, info

    # --- observación y helpers ---------------------------------------------
    def _dist_to_goal(self) -> float:
        _, py, _ = self.robot.get_position()
        return abs(self.params.goal_y - py)

    def _angle_to_goal(self) -> float:
        px, py, _ = self.robot.get_position()
        # Vector hacia la meta (plano y = goal_y); dirección a lo largo de y.
        dx = 0.0
        dy = self.params.goal_y - py
        target_heading = math.atan2(-dx, dy) if (dx or dy) else 0.0
        heading = self.robot.get_orientation()
        diff = target_heading - heading
        # Normalizar a [-pi, pi].
        return math.atan2(math.sin(diff), math.cos(diff))

    def _nearest_distance(self, nodes: Dict[str, Tuple[float, float]]) -> Optional[float]:
        if not nodes:
            return None
        px, py, _ = self.robot.get_position()
        return min(math.hypot(px - x, py - y) for (x, y) in nodes.values())

    def _get_obs(self) -> np.ndarray:
        dist = self._dist_to_goal()
        angle = self._angle_to_goal()
        heading = self.robot.get_orientation()
        nearest_obstacle = self._nearest_distance(self.robot.find_barriers())
        nearest_other = self._nearest_distance(self.robot.find_others())

        max_dist = self.params.max_dist
        obs = np.array(
            [
                min(dist / max_dist, 1.0),
                max(-1.0, min(angle / math.pi, 1.0)),
                max(-1.0, min(heading / math.pi, 1.0)),
                1.0 if nearest_obstacle is None else min(nearest_obstacle / max_dist, 1.0),
                1.0 if nearest_other is None else min(nearest_other / max_dist, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _base_info(self) -> dict:
        px, py, _ = self.robot.get_position()
        return {"robot_name": self.robot_name, "x": float(px), "y": float(py), "steps": self._steps}
