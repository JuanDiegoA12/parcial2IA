"""Detector de ganador de la carrera DARwIn-OP (modo race).

La pista es OVALADA y el inicio coincide con el fin, así que ganar NO es cruzar
una línea fija: el ganador es el PRIMER robot que COMPLETA la VUELTA de
checkpoints ordenados (cp0 -> cp1 -> ... -> cp7 -> cp0). El detector mantiene un
CheckpointTracker por robot (la MISMA máquina de estados que usa el entorno de
entrenamiento), de modo que la recompensa del entorno y la victoria de la
carrera coinciden por construcción.

Este módulo no depende de Webots ni de SB3: recibe, tick a tick, las posiciones
de los robots y decide el ganador. Así la lógica es unit-testable contra el
backend mock de los tests (basta con alimentar posiciones o escanear el mundo
mock vía find_checkpoints / get_position).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

try:  # config como módulo del paquete o como módulo suelto (pythonpath).
    from . import config
    from .race_env import CheckpointTracker
except ImportError:  # pragma: no cover - ruta usada bajo pythonpath plano
    import config
    from race_env import CheckpointTracker


def _ordered_from_coords(
    coords: Dict[int, Tuple[float, float]],
    order: Sequence[int],
    goal_y: float,
) -> List[Tuple[float, float]]:
    """Construye la secuencia ordenada de objetivos desde un mapa índice->(x,y).

    cp0 se sintetiza en (0, goal_y); el resto sale de `coords` (o del fallback
    config.CHECKPOINT_COORDS si falta alguno).
    """
    targets: List[Tuple[float, float]] = []
    for index in order:
        if index == 0:
            targets.append((0.0, float(goal_y)))
        elif index in coords:
            targets.append(coords[index])
        elif index in config.CHECKPOINT_COORDS:
            targets.append(config.CHECKPOINT_COORDS[index])
    return targets


class RaceWinnerDetector:
    """Detecta el primer robot que COMPLETA la vuelta de checkpoints ordenados.

    Parameters
    ----------
    goal_y:
        Plano de la meta (y) usado para sintetizar cp0. Por defecto config.GOAL_Y.
    checkpoints:
        Secuencia ORDENADA de coordenadas de objetivos (cp1..cp7, cp0). Si es
        None se construye desde config.CHECKPOINT_COORDS + CHECKPOINT_ORDER.
    order:
        Orden de la vuelta (por defecto config.CHECKPOINT_ORDER).
    radius:
        Radio para considerar alcanzado un objetivo (CHECKPOINT_RADIUS).
    """

    def __init__(self, goal_y: float = config.GOAL_Y,
                 checkpoints: Optional[Sequence[Tuple[float, float]]] = None,
                 order: Sequence[int] = config.CHECKPOINT_ORDER,
                 radius: float = config.CHECKPOINT_RADIUS):
        self.goal_y = float(goal_y)
        self.order = tuple(order)
        self.radius = float(radius)
        if checkpoints is None:
            checkpoints = _ordered_from_coords(
                config.CHECKPOINT_COORDS, self.order, self.goal_y
            )
        self._targets: List[Tuple[float, float]] = [
            (float(x), float(y)) for (x, y) in checkpoints
        ]
        self.winner: Optional[str] = None
        self.finish_order: list = []  # nombres en orden de llegada
        self._trackers: Dict[str, CheckpointTracker] = {}

    def _tracker_for(self, name: str) -> CheckpointTracker:
        tracker = self._trackers.get(name)
        if tracker is None:
            tracker = CheckpointTracker(self._targets, radius=self.radius)
            self._trackers[name] = tracker
        return tracker

    def reset(self) -> None:
        """Reinicia el estado del detector para una nueva carrera."""
        self.winner = None
        self.finish_order = []
        self._trackers = {}

    def update(self, positions: Dict[str, Tuple[float, float]]) -> Optional[str]:
        """Procesa un tick con las posiciones (x, y) de cada robot por nombre.

        Alimenta la posición de cada robot a su CheckpointTracker y registra en
        orden de llegada a los que COMPLETAN la vuelta. Fija el ganador la
        primera vez que alguno la completa.

        Returns
        -------
        El nombre del ganador si ya está decidido, o None todavía.
        """
        completed: List[str] = []
        for name, xy in positions.items():
            if name in self.finish_order:
                continue
            tracker = self._tracker_for(name)
            result = tracker.update(float(xy[0]), float(xy[1]))
            if result["done"]:
                completed.append(name)
        # Orden determinista cuando varios completan en el mismo tick: por el
        # orden de inserción del dict (estable en Python 3.7+).
        for name in completed:
            if name not in self.finish_order:
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
        Plano de la meta (y) para sintetizar cp0.
    action_fn:
        Callable(name, robot) -> None que aplica una acción al robot en cada
        tick (p. ej. una política greedy). Si es None, todos avanzan recto
        (walk_forward), suficiente para una prueba determinista.

    Returns
    -------
    dict con 'winner' (nombre o None), 'finish_order' (lista) y 'steps'.
    """
    # Construir la secuencia ordenada de checkpoints desde el PRIMER robot
    # (misma fuente que el entorno: find_checkpoints en runtime, con fallback a
    # config). Así el detector y el entorno coinciden.
    checkpoints: Optional[Sequence[Tuple[float, float]]] = None
    for robot in robots.values():
        getter = getattr(robot, "ordered_checkpoints", None)
        if getter is not None:
            try:
                checkpoints = getter(order=config.CHECKPOINT_ORDER, goal_y=goal_y)
            except Exception:
                checkpoints = None
        break

    detector = RaceWinnerDetector(goal_y=goal_y, checkpoints=checkpoints)
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
