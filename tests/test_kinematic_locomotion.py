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


# --- Velocidad realista y bloqueo de colisión (modo kinematic refinado) ------

def test_walk_forward_advances_by_config_linear_step(fake_webots):
    # (a) El avance usa config.KINEMATIC_LINEAR_STEP (fuente única), NO 0.25.
    robot, _ = _make_robot(fake_webots, "ppo")
    x0, y0, z0 = robot.get_position()
    assert abs(robot.get_orientation()) < 1e-9  # heading 0
    robot.walk_forward()
    robot.tick()
    x1, y1, z1 = robot.get_position()
    assert y1 - y0 == pytest.approx(config.KINEMATIC_LINEAR_STEP)
    assert x1 == pytest.approx(x0)
    assert z1 == pytest.approx(z0)


def _place_self_behind(supervisor, target_xy, gap=0.2):
    """Reubica el nodo propio a `gap` metros en -y de `target_xy`, heading 0.

    A heading 0 el avance es +y, así que el robot mira hacia el obstáculo y el
    destino (y + KINEMATIC_LINEAR_STEP) cae dentro del radio de colisión.
    """
    node = supervisor.getSelf()
    bx, by = target_xy
    node.set_translation([bx, by - gap, config.KINEMATIC_STAND_Z])
    node.set_rotation([0.0, 0.0, 1.0, 0.0])


def test_barrier_ahead_blocks_forward(fake_webots):
    # (b) Una valla directamente delante, a < KINEMATIC_COLLISION_RADIUS del
    # destino, BLOQUEA el avance: la posición no cambia (no penetra la valla).
    robot, supervisor = _make_robot(fake_webots, "DQN")
    barrier = fake_webots.node("valla 1(9)").translation  # (-0.895, -5.974)
    _place_self_behind(supervisor, (barrier[0], barrier[1]), gap=0.2)
    x0, y0, _ = robot.get_position()
    robot.walk_forward()
    robot.tick()
    x1, y1, _ = robot.get_position()
    assert (x1, y1) == pytest.approx((x0, y0))
    # El robot no se ACERCÓ a la valla (el avance se canceló; no la penetra).
    dist0 = math.hypot(x0 - barrier[0], y0 - barrier[1])
    dist1 = math.hypot(x1 - barrier[0], y1 - barrier[1])
    assert dist1 == pytest.approx(dist0)


def test_turn_still_works_when_blocked(fake_webots):
    # (c) En la misma configuración bloqueada, el giro SÍ cambia el heading.
    robot, supervisor = _make_robot(fake_webots, "DQN")
    barrier = fake_webots.node("valla 1(9)").translation
    _place_self_behind(supervisor, (barrier[0], barrier[1]), gap=0.2)
    h0 = robot.get_orientation()
    robot.turn_left()
    robot.tick()
    h1 = robot.get_orientation()
    assert h1 - h0 == pytest.approx(config.KINEMATIC_ANGULAR_STEP)


def test_other_robot_ahead_blocks_forward(fake_webots):
    # (d) Otro robot directamente delante también bloquea el avance.
    robot, supervisor = _make_robot(fake_webots, "DQN")
    other = fake_webots.node("ppo").translation
    _place_self_behind(supervisor, (other[0], other[1]), gap=0.2)
    x0, y0, _ = robot.get_position()
    robot.walk_forward()
    robot.tick()
    x1, y1, _ = robot.get_position()
    assert (x1, y1) == pytest.approx((x0, y0))


def _make_robot_real_webots(fake_webots, name="DQN"):
    """Robot sobre un supervisor que NO expone queue_action (emula Webots real).

    En Webots real el backend no tiene queue_action, de modo que
    _apply_kinematic_motion escribe translation/rotation directamente (en vez de
    delegar en el modelo cinemático del mock). Envolvemos el FakeSupervisor para
    ocultar queue_action y así ejercitar ESE camino (incluido el pin de z).
    """
    inner = fake_webots.make_supervisor(self_name=name)

    class _NoQueueSupervisor:
        def __getattr__(self, attr):
            if attr == "queue_action":
                raise AttributeError(attr)
            return getattr(inner, attr)

    return DarwinRobot(_NoQueueSupervisor(), name), inner


def test_kinematic_translation_pins_stand_z_and_upright(fake_webots):
    # (e) Camino real-Webots: supervisor SIN queue_action. El arrastre fija
    # z = KINEMATIC_STAND_Z (aunque el torso parta de una z colapsada) y la
    # rotación queda como yaw puro (rot[0]==rot[1]==0, rot[2]==1).
    robot, inner = _make_robot_real_webots(fake_webots, "DQN")
    assert robot.locomotion.mode == "kinematic"
    node = inner.getSelf()
    x, y, _z = node.translation
    # Partir de una z colapsada para comprobar que el pin la corrige.
    node.set_translation([x, y, 0.075])
    node.set_rotation([0.0, 0.0, 1.0, 0.0])  # heading 0, yaw puro
    robot.walk_forward()
    robot.tick()
    x1, y1, z1 = robot.get_position()
    assert z1 == pytest.approx(config.KINEMATIC_STAND_Z)
    assert y1 - y == pytest.approx(config.KINEMATIC_LINEAR_STEP)
    rot = node.rotation
    assert rot[0] == pytest.approx(0.0)
    assert rot[1] == pytest.approx(0.0)
    assert rot[2] == pytest.approx(1.0)
