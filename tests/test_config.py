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


# --- Checkpoints de la vuelta (pista ovalada) -------------------------------

def test_checkpoint_constants_have_expected_defaults():
    assert config.CHECKPOINT_RADIUS == 1.0
    assert config.CHECKPOINT_PROGRESS_WEIGHT == 10.0
    assert config.CHECKPOINT_BONUS == 20.0
    assert config.OFF_TRACK_PENALTY_WEIGHT == 2.0
    assert config.OFF_TRACK_TOLERANCE == 1.0
    assert config.CHECKPOINT_ORDER == (1, 2, 3, 4, 5, 6, 7, 0)
    assert config.TOTAL_TARGETS == 8


def test_checkpoint_coords_fallback_matches_world():
    # cp0 sintetizado en la meta; cp1..cp7 replican el .wbt.
    assert config.CHECKPOINT_COORDS[0] == (0.0, config.GOAL_Y)
    assert config.CHECKPOINT_COORDS[1] == (8.79, 7.98)
    assert config.CHECKPOINT_COORDS[7] == (-8.5, 8.21)
    # Todos los índices de la vuelta están cubiertos por el fallback.
    assert set(config.CHECKPOINT_COORDS) == set(config.CHECKPOINT_ORDER)


def test_params_carry_new_checkpoint_fields():
    base = config.RaceParams()
    assert base.checkpoint_radius == config.CHECKPOINT_RADIUS
    assert base.checkpoint_progress_weight == config.CHECKPOINT_PROGRESS_WEIGHT
    assert base.checkpoint_bonus == config.CHECKPOINT_BONUS
    assert base.off_track_penalty_weight == config.OFF_TRACK_PENALTY_WEIGHT
    assert base.off_track_tolerance == config.OFF_TRACK_TOLERANCE
    assert tuple(base.checkpoint_order) == config.CHECKPOINT_ORDER
    assert base.checkpoint_coords[1] == config.CHECKPOINT_COORDS[1]
    # override no destructivo sobre un campo nuevo.
    modified = base.override(checkpoint_radius=2.5)
    assert modified.checkpoint_radius == 2.5
    assert base.checkpoint_radius == config.CHECKPOINT_RADIUS
