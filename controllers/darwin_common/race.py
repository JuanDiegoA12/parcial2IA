"""Detector de ganador de la carrera DARwIn-OP (modo race).

El ganador es el PRIMER robot que cruza la línea de meta (plano y = GOAL_Y).
Este módulo no depende de Webots ni de SB3: recibe, tick a tick, las posiciones
de los robots y decide el ganador. Así la lógica es unit-testable contra el
backend mock de los tests (basta con alimentar posiciones o escanear el mundo
mock vía find_others / get_position).

Convención del mundo (ver config): el eje de carrera es y y se gana al alcanzar
y >= GOAL_Y. El detector es direccional (cruzar hacia +y), consistente con la
recompensa del entorno (success cuando py >= goal_y).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:  # config como módulo del paquete o como módulo suelto (pythonpath).
    from . import config
except ImportError:  # pragma: no cover - ruta usada bajo pythonpath plano
    import config


class RaceWinnerDetector:
    """Detecta el primer robot que cruza la meta y = goal_y.

    Parameters
    ----------
    goal_y:
        Plano de la meta en el eje y. Por defecto config.GOAL_Y.
    """

    def __init__(self, goal_y: float = config.GOAL_Y):
        self.goal_y = float(goal_y)
        self.winner: Optional[str] = None
        self.finish_order: list = []  # nombres en orden de llegada

    def reset(self) -> None:
        """Reinicia el estado del detector para una nueva carrera."""
        self.winner = None
        self.finish_order = []

    def update(self, positions: Dict[str, Tuple[float, float]]) -> Optional[str]:
        """Procesa un tick con las posiciones (x, y) de cada robot por nombre.

        Registra en orden de llegada a los robots que cruzan la meta y fija el
        ganador la primera vez que alguno la cruza.

        Returns
        -------
        El nombre del ganador si ya está decidido, o None todavía.
        """
        # Orden determinista cuando varios cruzan en el mismo tick: el que está
        # más adelantado (mayor y) va primero.
        crossing = [
            (name, xy[1])
            for name, xy in positions.items()
            if name not in self.finish_order and xy[1] >= self.goal_y
        ]
        crossing.sort(key=lambda item: item[1], reverse=True)
        for name, _y in crossing:
            self.finish_order.append(name)
            if self.winner is None:
                self.winner = name
        return self.winner


def run_race_until_winner(robots: Dict[str, object], max_steps: int,
                          goal_y: float = config.GOAL_Y,
                          action_fn=None) -> Dict[str, object]:
    """Corre una carrera tick a tick hasta que haya un ganador o se agoten pasos.

    Parameters
    ----------
    robots:
        Mapa nombre -> DarwinRobot (cada uno sobre su backend, real o mock).
    max_steps:
        Límite de pasos de simulación.
    goal_y:
        Plano de la meta.
    action_fn:
        Callable(name, robot) -> None que aplica una acción al robot en cada
        tick (p. ej. una política greedy). Si es None, todos avanzan recto
        (walk_forward), suficiente para una prueba determinista.

    Returns
    -------
    dict con 'winner' (nombre o None), 'finish_order' (lista) y 'steps'.
    """
    detector = RaceWinnerDetector(goal_y=goal_y)
    if action_fn is None:
        def action_fn(_name, robot):  # noqa: ANN001 - helper local
            robot.walk_forward()

    steps = 0
    for steps in range(1, int(max_steps) + 1):
        positions: Dict[str, Tuple[float, float]] = {}
        for name, robot in robots.items():
            action_fn(name, robot)
            robot.tick()
            px, py, _ = robot.get_position()
            positions[name] = (px, py)
        if detector.update(positions) is not None:
            break

    return {
        "winner": detector.winner,
        "finish_order": list(detector.finish_order),
        "steps": steps,
    }
