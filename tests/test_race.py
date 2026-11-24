"""Tests del detector de ganador de la carrera (controllers/darwin_common/race.py).

Verifican que RaceWinnerDetector reporta el PRIMER robot en cruzar y=GOAL_Y,
tanto con posiciones alimentadas a mano como corriendo la carrera sobre el
backend mock con DarwinRobot real (fixture fake_webots).
"""

import config
import race
from darwin_robot import DarwinRobot


def test_detector_first_to_cross_wins():
    detector = race.RaceWinnerDetector(goal_y=9.0)
    # Nadie cruza aún.
    assert detector.update({"A": (0.0, 1.0), "B": (0.0, 2.0)}) is None
    # B cruza primero.
    assert detector.update({"A": (0.0, 8.5), "B": (0.0, 9.1)}) == "B"
    # El ganador no cambia aunque A cruce después.
    assert detector.update({"A": (0.0, 9.5), "B": (0.0, 10.0)}) == "B"
    assert detector.finish_order[0] == "B"


def test_detector_same_tick_most_advanced_wins():
    detector = race.RaceWinnerDetector(goal_y=9.0)
    # Ambos cruzan en el mismo tick; gana el más adelantado (mayor y).
    winner = detector.update({"A": (0.0, 9.2), "B": (0.0, 9.8)})
    assert winner == "B"
    assert detector.finish_order == ["B", "A"]


def test_detector_uses_config_goal_by_default():
    detector = race.RaceWinnerDetector()
    assert detector.goal_y == config.GOAL_Y


def test_run_race_until_winner_on_mock(fake_webots):
    # 'agente propio' empieza en y=11.99 (ya pasada la meta y=9): gana de
    # inmediato; 'actor - critic' empieza en y=6.04 (antes de la meta).
    robots = {
        "agente propio": DarwinRobot(
            fake_webots.make_supervisor(self_name="agente propio"), "agente propio"
        ),
        "actor - critic": DarwinRobot(
            fake_webots.make_supervisor(self_name="actor - critic"), "actor - critic"
        ),
    }
    result = race.run_race_until_winner(robots, max_steps=50, goal_y=9.0)
    assert result["winner"] == "agente propio"
    assert result["finish_order"][0] == "agente propio"


def test_run_race_until_winner_progresses(fake_webots):
    # Un robot que arranca antes de la meta debe cruzarla avanzando recto.
    backend = fake_webots.make_supervisor(self_name="actor - critic")
    robots = {"actor - critic": DarwinRobot(backend, "actor - critic")}
    # Desde y=6.04 con WALK_DELTA=0.25, hacen falta ~12 pasos para llegar a y=9.
    result = race.run_race_until_winner(robots, max_steps=50, goal_y=9.0)
    assert result["winner"] == "actor - critic"
    assert result["steps"] <= 50
