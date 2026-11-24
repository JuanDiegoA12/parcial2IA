"""Tests de DarwinRobot contra el backend mock (fake_webots)."""

import math
import sys

import pytest

import config
from darwin_robot import DarwinRobot


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


def test_locomotion_fallback_triggers_on_managers_failure(fake_webots):
    # Forzar fallo del import de managers: lo reemplazamos por un módulo roto.
    import types

    broken = types.ModuleType("managers")
    sys.modules["managers"] = broken  # sin las clases -> ImportError interno
    try:
        with pytest.warns(RuntimeWarning):
            robot, _ = _make_robot(fake_webots, "DQN")
        assert robot.locomotion.available is False
        # El stub no lanza excepción al caminar.
        robot.walk_forward()
        robot.tick()
    finally:
        del sys.modules["managers"]
