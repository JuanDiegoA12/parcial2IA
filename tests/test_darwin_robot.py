"""Tests de DarwinRobot contra el backend mock (fake_webots)."""

import math

import pytest

import config
from darwin_robot import DarwinRobot, _GaitEngine, _clamp_joint


def _make_robot(fake_webots, name="DQN"):
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
    robot, supervisor = _make_robot(fake_webots, "DQN")
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
    lo, hi = config.JOINT_LIMITS["LegLowerR"]  # (-2.25, 2.25)
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
    robot, _ = _make_robot(fake_webots, "ppo")
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
    robot, supervisor = _make_robot(fake_webots, "DQN")
    assert robot.is_fallen() is False
    # Bajar el torso por debajo de FALL_HEIGHT_Z sin tocar el acelerómetro.
    node = supervisor.getSelf()
    x, y, _z = node.translation
    node.set_translation([x, y, config.FALL_HEIGHT_Z - 0.05])
    assert robot.is_fallen() is True


def test_is_fallen_true_on_strong_tilt(fake_webots):
    robot, supervisor = _make_robot(fake_webots, "DQN")
    supervisor.force_tilt(config.FALL_TILT_THRESHOLD + 0.5)
    assert robot.is_fallen() is True
