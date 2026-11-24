"""Tests de DarwinRobot contra el backend mock (fake_webots)."""

import math
import os

import pytest

import config
from darwin_robot import DarwinRobot, _GaitEngine, _clamp_joint


@pytest.fixture(autouse=True)
def _restore_locomotion_mode():
    """Restaura la variable de entorno LOCOMOTION_MODE tras cada test.

    Algunos tests fijan el modo ('gait') antes de construir el robot; este
    fixture evita que esa selección se filtre a otros tests.
    """
    saved = os.environ.get("LOCOMOTION_MODE")
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("LOCOMOTION_MODE", None)
        else:
            os.environ["LOCOMOTION_MODE"] = saved


def _make_robot(fake_webots, name="DQN", mode=None):
    # El modo de locomoción se resuelve al CONSTRUIR el robot, así que si un
    # test necesita un modo concreto (p.ej. 'gait' para las señales de caída y
    # el avance de fase del gait, que el modo kinematic por diseño desactiva)
    # debe fijar LOCOMOTION_MODE antes de instanciar DarwinRobot.
    if mode is not None:
        os.environ["LOCOMOTION_MODE"] = mode
    supervisor = fake_webots.make_supervisor(self_name=name)
    return DarwinRobot(supervisor, name), supervisor


def test_get_position_reads_start(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    x, y, z = robot.get_position()
    assert (round(x, 2), round(y, 2)) == (0.11, 9.95)


def test_walk_forward_advances_toward_goal(fake_webots):
    robot, _ = _make_robot(fake_webots, "ppo")
    _, y0, _ = robot.get_position()
    robot.walk_forward()
    robot.tick()
    _, y1, _ = robot.get_position()
    assert y1 > y0  # heading 0 avanza en +y


def test_turn_changes_heading(fake_webots):
    robot, _ = _make_robot(fake_webots, "ppo")
    h0 = robot.get_orientation()
    robot.turn_left()
    robot.tick()
    h_left = robot.get_orientation()
    assert h_left > h0
    robot.turn_right()
    robot.tick()
    robot.turn_right()
    robot.tick()
    assert robot.get_orientation() < h_left


def test_reset_pose_sets_translation(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    robot.reset_pose(1.5, -2.5)
    x, y, _ = robot.get_position()
    assert (round(x, 2), round(y, 2)) == (1.5, -2.5)
    assert abs(robot.get_orientation()) < 1e-9


def test_is_fallen_threshold(fake_webots):
    # La detección de caída es una señal de Plan A (gait); kinematic la
    # desactiva por diseño, así que este test se ejecuta en modo gait.
    robot, supervisor = _make_robot(fake_webots, "DQN", mode="gait")
    assert robot.is_fallen() is False
    supervisor.force_tilt(config.FALL_TILT_THRESHOLD + 0.3)
    assert robot.is_fallen() is True
    supervisor.force_tilt(0.1)
    assert robot.is_fallen() is False


def test_find_others_excludes_self_and_finds_robots(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    others = robot.find_others()
    assert "DQN" not in others
    assert {"ppo", "actor - critic", "agente propio"}.issubset(set(others))


def test_find_barriers_returns_twelve(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    barriers = robot.find_barriers()
    assert len(barriers) == 12
    assert all(name.startswith("valla") for name in barriers)


def test_is_supervisor_true_when_self_node_resolves(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    assert robot.is_supervisor() is True


def test_non_supervisor_robot_is_tolerated(fake_webots):
    # Si el robot NO es Supervisor, getSelf() devuelve None (en Webots real,
    # además registra "illegal call to wb_supervisor_node_get_self()"). El
    # robot debe construirse sin reventar, is_supervisor() == False, y las
    # llamadas que necesitan el nodo propio deben lanzar un RuntimeError con el
    # mensaje accionable en vez de un AttributeError sobre NoneType.
    supervisor = fake_webots.make_supervisor(self_name="DQN")
    supervisor.getSelf = lambda: None
    robot = DarwinRobot(supervisor, "DQN")
    assert robot.is_supervisor() is False
    with pytest.raises(RuntimeError, match="supervisor TRUE"):
        robot.get_position()
    with pytest.raises(RuntimeError, match="supervisor TRUE"):
        robot.get_orientation()
    with pytest.raises(RuntimeError, match="supervisor TRUE"):
        robot.reset_pose(1.0, 2.0)


def test_non_supervisor_robot_tolerates_getself_raising(fake_webots):
    # En Webots real getSelf() puede LANZAR en un controlador 'Robot'; la
    # construcción debe tolerarlo y marcar is_supervisor() == False.
    supervisor = fake_webots.make_supervisor(self_name="DQN")

    def _raise():
        raise RuntimeError("illegal call to wb_supervisor_node_get_self()")

    supervisor.getSelf = _raise
    robot = DarwinRobot(supervisor, "DQN")
    assert robot.is_supervisor() is False


def test_locomotion_noop_when_no_motors_resolve(fake_webots):
    # Backend sin getMotor/getDevice: ningún motor resuelve -> modo no-op con
    # warning, pero la locomoción sigue funcionando sin lanzar excepción.
    supervisor = fake_webots.make_supervisor(self_name="DQN")
    # Ocultar los getters de motores para simular un backend sin motores.
    supervisor.getMotor = None
    supervisor.getDevice = None
    with pytest.warns(RuntimeWarning):
        robot = DarwinRobot(supervisor, "DQN")
    assert robot.locomotion.available is False
    # Sin motores, caminar y avanzar un tick no debe lanzar excepción.
    robot.walk_forward()
    robot.tick()


# --- Tests del gait sinusoidal (_GaitEngine) --------------------------------

def test_clamp_joint_keeps_margin():
    lo, hi = config.JOINT_LIMITS["LegLowerR"]  # real PROTO: (-0.02, 2.25)
    margin = config.JOINT_LIMIT_MARGIN
    # Pedir justo el tope -> se recorta a tope-margen.
    assert _clamp_joint("LegLowerR", hi + 10.0) == pytest.approx(hi - margin)
    assert _clamp_joint("LegLowerR", lo - 10.0) == pytest.approx(lo + margin)
    # Un valor interior pasa sin cambios.
    assert _clamp_joint("LegLowerR", 0.3) == pytest.approx(0.3)


def test_gait_targets_stay_within_safe_limits():
    engine = _GaitEngine()
    engine.set_amplitudes(config.GAIT_X_AMPLITUDE, config.GAIT_A_AMPLITUDE)
    margin = config.JOINT_LIMIT_MARGIN
    # Barremos un ciclo completo de fase.
    for i in range(60):
        phase = 2.0 * math.pi * i / 60.0
        targets = engine.compute_targets(phase)
        for name, angle in targets.items():
            lo, hi = config.JOINT_LIMITS[name]
            assert lo + margin - 1e-9 <= angle <= hi - margin + 1e-9, (name, angle)


def test_gait_phase_advances_on_tick(fake_webots):
    # El avance de fase del gait sólo ocurre en modo gait (Plan A).
    robot, _ = _make_robot(fake_webots, "ppo", mode="gait")
    p0 = robot.locomotion.engine.phase
    robot.walk_forward()
    robot.tick()
    assert robot.locomotion.engine.phase > p0


def test_left_and_right_legs_are_180_out_of_phase():
    engine = _GaitEngine()
    engine.set_amplitudes(config.GAIT_X_AMPLITUDE, 0.0)
    base = dict(config.STANDING_POSE)
    # La contribución de cadera de cada pierna respecto a la pose nominal.
    t = engine.compute_targets(0.0)
    hip_r = t["LegUpperR"] - base["LegUpperR"]
    hip_l = t["LegUpperL"] - base["LegUpperL"]
    # A fase 0: sin(0)=0 para la derecha, sin(pi)=0 para la izquierda.
    assert hip_r == pytest.approx(0.0, abs=1e-6)
    assert hip_l == pytest.approx(0.0, abs=1e-6)
    # A fase pi/2 las contribuciones deben tener signo opuesto (180 de desfase).
    t2 = engine.compute_targets(math.pi / 2.0)
    hip_r2 = t2["LegUpperR"] - base["LegUpperR"]
    hip_l2 = t2["LegUpperL"] - base["LegUpperL"]
    assert hip_r2 * hip_l2 < 0


def test_stand_sets_nominal_pose():
    engine = _GaitEngine()
    engine.set_amplitudes(0.0, 0.0)
    targets = engine.compute_targets(0.0)
    # Sin amplitudes y a fase 0 el gait equivale a la pose de pie (recortada),
    # salvo el término de sway lateral (cos(0)=1) sobre pelvis/tobillos.
    for name in ("LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL"):
        assert targets[name] == pytest.approx(_clamp_joint(name, config.STANDING_POSE[name]))


def test_actions_set_x_and_a_amplitudes(fake_webots):
    robot, _ = _make_robot(fake_webots, "ppo")
    engine = robot.locomotion.engine

    robot.walk_forward()
    assert engine.x_amplitude == pytest.approx(config.GAIT_X_AMPLITUDE)
    assert engine.a_amplitude == pytest.approx(0.0)

    robot.turn_left()
    assert engine.x_amplitude == pytest.approx(0.0)
    assert engine.a_amplitude == pytest.approx(config.GAIT_A_AMPLITUDE)

    robot.turn_right()
    assert engine.a_amplitude == pytest.approx(-config.GAIT_A_AMPLITUDE)

    robot.stand()
    assert engine.x_amplitude == pytest.approx(0.0)
    assert engine.a_amplitude == pytest.approx(0.0)


def test_is_fallen_true_on_low_height(fake_webots):
    robot, supervisor = _make_robot(fake_webots, "DQN", mode="gait")
    assert robot.is_fallen() is False
    # Bajar el torso por debajo de FALL_HEIGHT_Z sin tocar el acelerómetro.
    node = supervisor.getSelf()
    x, y, _z = node.translation
    node.set_translation([x, y, config.FALL_HEIGHT_Z - 0.05])
    assert robot.is_fallen() is True


def test_is_fallen_true_on_strong_tilt(fake_webots):
    robot, supervisor = _make_robot(fake_webots, "DQN", mode="gait")
    supervisor.force_tilt(config.FALL_TILT_THRESHOLD + 0.5)
    assert robot.is_fallen() is True


# --- Límites de motor leídos en runtime (fix A) -----------------------------

def test_backend_reads_runtime_motor_limits(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    limits = robot.locomotion.motor_limits
    # Todos los motores resueltos tienen límites leídos del FakeMotor.
    assert set(limits) == set(config.MOTOR_NAMES)
    # Y coinciden con los que expone el motor (= config.JOINT_LIMITS del mock).
    assert limits["LegUpperR"] == config.JOINT_LIMITS["LegUpperR"]


def test_clamp_respects_runtime_limits(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    backend = robot.locomotion
    lo, hi = backend.motor_limits["LegUpperR"]  # real: (-1.77, 0.45)
    margin = config.JOINT_LIMIT_MARGIN
    # Pedir por encima del tope real -> recortado a max-margen (nunca el tope).
    assert backend.clamp("LegUpperR", hi + 5.0) == pytest.approx(hi - margin)
    assert backend.clamp("LegUpperR", lo - 5.0) == pytest.approx(lo + margin)
    assert backend.clamp("LegUpperR", hi + 5.0) < hi  # nunca exactamente el tope


def test_apply_targets_never_exceeds_runtime_limits(fake_webots):
    robot, _ = _make_robot(fake_webots, "DQN")
    backend = robot.locomotion
    margin = config.JOINT_LIMIT_MARGIN
    # Comandar ángulos deliberadamente fuera de rango a todos los motores.
    backend.apply_targets({name: 100.0 for name in config.MOTOR_NAMES})
    for name, motor in backend.motors.items():
        lo, hi = backend.motor_limits[name]
        assert lo + margin - 1e-9 <= motor.position <= hi - margin + 1e-9, name


# --- Pose de pie dentro de los límites (fix B) ------------------------------

def test_standing_pose_within_fallback_limits_with_margin():
    margin = config.JOINT_LIMIT_MARGIN
    for name, angle in config.STANDING_POSE.items():
        limits = config.JOINT_LIMITS.get(name)
        assert limits is not None, name
        lo, hi = limits
        assert lo + margin <= angle <= hi - margin, (name, angle, limits)


# --- Persistencia del acelerómetro en la caída (fix C) ----------------------

def test_accel_fall_requires_persistence(fake_webots):
    robot, supervisor = _make_robot(fake_webots, "DQN", mode="gait")
    # Sacar SOLO el acelerómetro de banda, sin tocar altura ni orientación.
    out_of_band = (config.ACCEL_CENTER + config.ACCEL_FALL_TOLERANCE + 50.0,
                   config.ACCEL_CENTER, config.ACCEL_CENTER)
    # Una única lectura fuera de banda NO declara caída.
    supervisor._accel.set_values(out_of_band)
    assert robot.is_fallen() is False
    # Tras FALL_ACCEL_PERSIST_STEPS ticks consecutivos fuera de banda, SÍ.
    for _ in range(config.FALL_ACCEL_PERSIST_STEPS):
        result = robot.is_fallen()
    assert result is True


def test_accel_out_of_band_counter_resets_on_normal_reading(fake_webots):
    robot, supervisor = _make_robot(fake_webots, "DQN", mode="gait")
    out_of_band = (config.ACCEL_CENTER + config.ACCEL_FALL_TOLERANCE + 50.0,
                   config.ACCEL_CENTER, config.ACCEL_CENTER)
    supervisor._accel.set_values(out_of_band)
    for _ in range(config.FALL_ACCEL_PERSIST_STEPS - 1):
        robot.is_fallen()
    # Una lectura dentro de banda reinicia el contador.
    supervisor._accel.set_values((config.ACCEL_CENTER,) * 3)
    assert robot.is_fallen() is False
    assert robot._accel_out_of_band == 0


# --- Umbral de altura (fix C) -----------------------------------------------

def test_is_fallen_false_standing_true_collapsed(fake_webots):
    robot, supervisor = _make_robot(fake_webots, "DQN", mode="gait")
    node = supervisor.getSelf()
    x, y, _z = node.translation
    # De pie / levemente agachado (~0.24): NO caído.
    node.set_translation([x, y, 0.24])
    assert robot.is_fallen() is False
    # Colapsado (~0.075): caído por la rama de altura.
    node.set_translation([x, y, 0.075])
    assert robot.is_fallen() is True
