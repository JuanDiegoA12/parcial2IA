"""Tests de config.py: constantes del mundo y parámetros sobre-escribibles."""

import config


def test_action_map_has_exactly_three_entries():
    assert len(config.ACTION_MAP) == 3
    assert set(config.ACTION_MAP.values()) == {"walk_forward", "turn_left", "turn_right"}


def test_start_positions_cover_all_four_robots():
    expected = {"DQN", "ppo", "actor - critic", "agente propio"}
    assert expected.issubset(set(config.START_POSITIONS))


def test_goal_and_start_values_match_world():
    assert config.GOAL_Y == 9.0
    assert config.START_POSITIONS["DQN"] == (0.11, 9.95, 0.24)
    assert config.START_POSITIONS["agente propio"] == (0.21, 11.99, 0.34)


def test_params_override_is_non_destructive():
    base = config.RaceParams()
    modified = base.override(goal_y=15.0, max_episode_steps=50)
    assert modified.goal_y == 15.0
    assert modified.max_episode_steps == 50
    # El original no cambia.
    assert base.goal_y == config.GOAL_Y
    assert base.max_episode_steps == config.MAX_EPISODE_STEPS
