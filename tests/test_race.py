"""Tests del detector de ganador de la carrera (controllers/darwin_common/race.py).

La pista es ovalada y el inicio coincide con el fin, así que ganar = COMPLETAR
la VUELTA de checkpoints ordenados (cp0 -> cp1 -> ... -> cp7 -> cp0), NO cruzar
una línea fija. Verifican que RaceWinnerDetector reporta el PRIMER robot que
cierra la vuelta, tanto con posiciones alimentadas a mano como corriendo la
carrera sobre el backend mock con DarwinRobot real (fixture fake_webots).
"""

import config
import race
from darwin_robot import DarwinRobot


def _ordered_targets(goal_y=config.GOAL_Y):
    """Secuencia ordenada de objetivos (cp1..cp7, cp0) desde el fallback."""
    targets = []
    for index in config.CHECKPOINT_ORDER:
        if index == 0:
            targets.append((0.0, goal_y))
        else:
            targets.append(config.CHECKPOINT_COORDS[index])
    return targets


def _drive_full_lap(detector, name, extra_robots=None):
    """Alimenta al detector la secuencia completa de la vuelta para `name`.

    Opcionalmente `extra_robots` es un dict name->posición fija que se incluye
    en cada tick (robots que NO completan la vuelta).
    """
    extra_robots = extra_robots or {}
    targets = _ordered_targets(detector.goal_y)
    winner = None
    for (tx, ty) in targets:
        positions = {name: (tx, ty)}
        positions.update(extra_robots)
        winner = detector.update(positions)
    return winner


def test_detector_first_to_complete_lap_wins():
    detector = race.RaceWinnerDetector(goal_y=9.0)
    # Estar en la meta cp0 sin recorrer cp1..cp7 NO gana.
    assert detector.update({"A": (0.0, 9.0), "B": (0.0, 9.1)}) is None
    # B completa la vuelta entera: gana.
    winner = _drive_full_lap(detector, "B", extra_robots={"A": (0.0, 1.0)})
    assert winner == "B"
    assert detector.finish_order[0] == "B"


def test_detector_crossing_goal_without_lap_does_not_win():
    detector = race.RaceWinnerDetector(goal_y=9.0)
    # Un robot parado sobre la meta por muchos ticks nunca gana sin la vuelta.
    for _ in range(20):
        assert detector.update({"A": (0.0, 9.0)}) is None
    assert detector.winner is None


def test_detector_uses_config_goal_by_default():
    detector = race.RaceWinnerDetector()
    assert detector.goal_y == config.GOAL_Y


def test_run_race_until_winner_on_mock(fake_webots):
    # Conducir a 'actor - critic' por toda la vuelta con una action_fn que lo
    # teletransporta al siguiente objetivo cada tick; 'agente propio' se queda
    # quieto en su inicio (no completa la vuelta).
    driven = DarwinRobot(
        fake_webots.make_supervisor(self_name="actor - critic"), "actor - critic"
    )
    idle = DarwinRobot(
        fake_webots.make_supervisor(self_name="agente propio"), "agente propio"
    )
    robots = {"actor - critic": driven, "agente propio": idle}

    targets = _ordered_targets()
    state = {"i": 0}

    def action_fn(name, robot):
        if name != "actor - critic":
            return  # 'agente propio' no se mueve
        i = min(state["i"], len(targets) - 1)
        tx, ty = targets[i]
        robot.backend.getSelf().set_translation([tx, ty, 0.34])
        state["i"] += 1

    result = race.run_race_until_winner(
        robots, max_steps=50, goal_y=9.0, action_fn=action_fn
    )
    assert result["winner"] == "actor - critic"
    assert result["finish_order"][0] == "actor - critic"


def test_run_race_until_winner_progresses(fake_webots):
    # Un único robot conducido por la vuelta completa debe ganar.
    backend = fake_webots.make_supervisor(self_name="actor - critic")
    robot = DarwinRobot(backend, "actor - critic")
    robots = {"actor - critic": robot}

    targets = _ordered_targets()
    state = {"i": 0}

    def action_fn(_name, r):
        i = min(state["i"], len(targets) - 1)
        tx, ty = targets[i]
        r.backend.getSelf().set_translation([tx, ty, 0.34])
        state["i"] += 1

    result = race.run_race_until_winner(
        robots, max_steps=50, goal_y=9.0, action_fn=action_fn
    )
    assert result["winner"] == "actor - critic"
    assert result["steps"] <= 50
