"""Tests del modo de locomoción cinemático (Plan B) contra el backend mock.

Plan B desliza al robot DE PIE vía Supervisor (arrastre tipo "tortuga"
diferencial) en vez de usar el gait sinusoidal físico (Plan A). El modo se
selecciona con la variable de entorno LOCOMOTION_MODE, que debe fijarse ANTES
de construir el robot (el modo se resuelve en _LocomotionBackend.__init__).

Reutilizamos la convención exacta del mock (tests/conftest.py): heading 0 mira
a +y; avance = (-sin(theta)*d, +cos(theta)*d, 0); girar suma +/- el paso
angular al ángulo en z. Como KINEMATIC_LINEAR_STEP = WALK_DELTA y
KINEMATIC_ANGULAR_STEP = TURN_DELTA, las magnitudes bajo el mock coinciden con
las constantes de config, así que las aserciones son deterministas.
"""

import math

import pytest

import config
from darwin_robot import DarwinRobot


@pytest.fixture(autouse=True)
def _kinematic_mode(monkeypatch):
    """Por defecto estos tests corren en modo kinematic (Plan B)."""
    monkeypatch.setenv("LOCOMOTION_MODE", "kinematic")


def _make_robot(fake_webots, name="DQN"):
    supervisor = fake_webots.make_supervisor(self_name=name)
    return DarwinRobot(supervisor, name), supervisor


def test_mode_resolves_to_kinematic(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    assert robot.locomotion.mode == "kinematic"


def test_walk_forward_advances_along_plus_y_at_heading_zero(fake_webots):
    robot, _ = _make_robot(fake_webots, "ppo")
    x0, y0, z0 = robot.get_position()
    assert abs(robot.get_orientation()) < 1e-9  # heading 0
    robot.walk_forward()
    robot.tick()
    x1, y1, z1 = robot.get_position()
    # Avanza KINEMATIC_LINEAR_STEP en +y; x y z no cambian a heading 0.
    assert y1 - y0 == pytest.approx(config.KINEMATIC_LINEAR_STEP)
    assert x1 == pytest.approx(x0)
    assert z1 == pytest.approx(z0)


def test_turn_left_increases_heading_by_angular_step(fake_webots):
    robot, _ = _make_robot(fake_webots, "ppo")
    h0 = robot.get_orientation()
    robot.turn_left()
    robot.tick()
    h1 = robot.get_orientation()
    assert h1 - h0 == pytest.approx(config.KINEMATIC_ANGULAR_STEP)


def test_turn_right_decreases_heading_by_angular_step(fake_webots):
    robot, _ = _make_robot(fake_webots, "ppo")
    h0 = robot.get_orientation()
    robot.turn_right()
    robot.tick()
    h1 = robot.get_orientation()
    assert h1 - h0 == pytest.approx(-config.KINEMATIC_ANGULAR_STEP)


def test_walk_forward_after_turn_follows_heading(fake_webots):
    # Tras girar, el avance debe seguir el nuevo heading (convención del mock).
    robot, _ = _make_robot(fake_webots, "ppo")
    x0, y0, _ = robot.get_position()
    robot.turn_left()
    robot.tick()
    theta = robot.get_orientation()
    robot.walk_forward()
    robot.tick()
    x1, y1, _ = robot.get_position()
    d = config.KINEMATIC_LINEAR_STEP
    assert x1 - x0 == pytest.approx(-math.sin(theta) * d)
    assert y1 - y0 == pytest.approx(math.cos(theta) * d)


def test_is_fallen_false_on_strong_tilt_in_kinematic_mode(fake_webots):
    # En kinematic la caída está desactivada aun con el acelerómetro fuera de
    # banda, orientación volcada y altura por debajo del umbral.
    robot, supervisor = _make_robot(fake_webots, "DQN")
    supervisor.force_tilt(config.FALL_TILT_THRESHOLD + 0.5)
    assert robot.is_fallen() is False


def test_is_fallen_false_on_low_height_in_kinematic_mode(fake_webots):
    robot, supervisor = _make_robot(fake_webots, "DQN")
    node = supervisor.getSelf()
    x, y, _z = node.translation
    node.set_translation([x, y, config.FALL_HEIGHT_Z - 0.05])
    assert robot.is_fallen() is False


def test_gait_mode_still_detects_fall(fake_webots, monkeypatch):
    # Contraprueba: en modo gait la caída SÍ se detecta (el ruteo por modo
    # funciona en ambos sentidos).
    monkeypatch.setenv("LOCOMOTION_MODE", "gait")
    robot, supervisor = _make_robot(fake_webots, "DQN")
    assert robot.locomotion.mode == "gait"
    supervisor.force_tilt(config.FALL_TILT_THRESHOLD + 0.5)
    assert robot.is_fallen() is True


def test_reset_pose_clears_pending_kinematic_action(fake_webots):
    # reset_pose debe limpiar cualquier arrastre pendiente al iniciar episodio.
    robot, _ = _make_robot(fake_webots, "ppo")
    robot.walk_forward()
    assert robot.locomotion._kin_action == "walk_forward"
    robot.reset_pose(1.0, 2.0)
    assert robot.locomotion._kin_action is None
