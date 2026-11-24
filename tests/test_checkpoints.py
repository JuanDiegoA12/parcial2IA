"""Tests de la máquina de estados de la vuelta (CheckpointTracker) y del
penalizador off-track, más la integración en DarwinRaceEnv.

CheckpointTracker es Python puro (sin Webots): se prueba alimentándole
posiciones. La parte de penalización off-track en el entorno usa el backend
mock (fixture fake_webots).
"""

import math

import pytest

import config
from race_env import CheckpointTracker, DarwinRaceEnv, _segment_distance


def _ordered_targets():
    """Secuencia ordenada de objetivos desde el fallback de config."""
    targets = []
    for index in config.CHECKPOINT_ORDER:
        if index == 0:
            targets.append((0.0, config.GOAL_Y))
        else:
            targets.append(config.CHECKPOINT_COORDS[index])
    return targets


def test_advancing_in_order_increases_reached_count():
    targets = _ordered_targets()
    tracker = CheckpointTracker(targets, radius=1.0)
    assert tracker.reached_count == 0
    # Teletransportar exactamente a cada objetivo en orden: cada uno avanza.
    for i, (tx, ty) in enumerate(targets[:-1], start=1):
        result = tracker.update(tx, ty)
        assert result["reached"] is True
        assert result["reached_count"] == i
        assert result["done"] is False


def test_reaching_later_checkpoint_early_is_not_credited():
    targets = _ordered_targets()
    tracker = CheckpointTracker(targets, radius=1.0)
    # Ir directo al objetivo 3 (índice 2) sin pasar por 1 y 2: NO cuenta.
    tx, ty = targets[2]
    result = tracker.update(tx, ty)
    assert result["reached"] is False
    assert result["reached_count"] == 0
    # El objetivo actual sigue siendo el primero (cp1).
    assert tracker.current_target() == pytest.approx(targets[0])


def test_done_only_after_final_return_to_cp0():
    targets = _ordered_targets()
    tracker = CheckpointTracker(targets, radius=1.0)
    # Alcanzar cp1..cp7 (todos menos el último, que es cp0).
    for tx, ty in targets[:-1]:
        res = tracker.update(tx, ty)
        assert res["done"] is False
    assert tracker.current_target() == pytest.approx(targets[-1])  # cp0
    # Volver a cp0 cierra la vuelta.
    final = tracker.update(targets[-1][0], targets[-1][1])
    assert final["reached"] is True
    assert final["done"] is True
    assert tracker.current_target() is None


def test_progress_is_per_step_distance_reduction():
    # Un solo objetivo para aislar el progreso: cp en (0, 10).
    tracker = CheckpointTracker([(0.0, 10.0)], radius=0.5)
    tracker.update(0.0, 0.0)   # prev_dist = 10
    result = tracker.update(0.0, 2.0)  # dist = 8 -> progreso = 2
    assert result["progress"] == pytest.approx(2.0)
    # Alejarse da progreso negativo.
    result = tracker.update(0.0, 1.0)  # dist = 9 -> progreso = -1
    assert result["progress"] == pytest.approx(-1.0)


def test_segment_distance_zero_on_line_positive_off_line():
    # Segmento de (0,0) a (0,10): sobre la línea d=0, fuera d>0.
    assert _segment_distance(0.0, 5.0, 0.0, 0.0, 0.0, 10.0) == pytest.approx(0.0)
    assert _segment_distance(3.0, 5.0, 0.0, 0.0, 0.0, 10.0) == pytest.approx(3.0)
    # Proyección recortada a los extremos del segmento.
    assert _segment_distance(0.0, 20.0, 0.0, 0.0, 0.0, 10.0) == pytest.approx(10.0)
    # Segmento degenerado (A == B): distancia punto-punto a A.
    # (4,5) a (1,1) = hypot(3,4) = 5.
    assert _segment_distance(4.0, 5.0, 1.0, 1.0, 1.0, 1.0) == pytest.approx(5.0)


def test_current_segment_uses_predecessor():
    targets = _ordered_targets()
    tracker = CheckpointTracker(targets, radius=1.0)
    # Primer tramo: origen (cp0) -> cp1.
    seg = tracker.current_segment()
    assert seg[0] == pytest.approx((0.0, config.GOAL_Y))
    assert seg[1] == pytest.approx(targets[0])
    # Tras alcanzar cp1, el tramo es cp1 -> cp2.
    tracker.update(targets[0][0], targets[0][1])
    seg = tracker.current_segment()
    assert seg[0] == pytest.approx(targets[0])
    assert seg[1] == pytest.approx(targets[1])


def test_finish_point_is_cp0():
    targets = _ordered_targets()
    tracker = CheckpointTracker(targets, radius=1.0)
    # cp0 (cierre de vuelta) es el último objetivo del orden canónico.
    assert tracker.finish_point() == pytest.approx((0.0, config.GOAL_Y))


# --- Integración: observación y penalización off-track en el entorno --------

def test_obs_lap_progress_and_dist_to_finish_are_distinct(fake_webots):
    """obs[3] (lap_progress) y obs[4] (dist_to_finish) NO son idénticos.

    En la pose de arranque lap_progress=0 (ningún checkpoint alcanzado) pero la
    distancia normalizada al cierre de vuelta (cp0) es >0 porque el robot no
    arranca exactamente sobre cp0; por tanto ambas señales difieren.
    """
    import os
    os.environ["LOCOMOTION_MODE"] = "kinematic"
    try:
        supervisor = fake_webots.make_supervisor(self_name="ppo")
        env = DarwinRaceEnv(supervisor, robot_name="ppo")
        obs, _ = env.reset()
        assert obs.shape == (7,)
        # lap_progress en el arranque: ningún objetivo alcanzado todavía.
        assert obs[3] == pytest.approx(0.0)
        # dist_to_finish normalizada: estrictamente >0 y != lap_progress.
        assert obs[4] > 0.0
        assert obs[4] != pytest.approx(obs[3])
        # El valor coincide con la distancia a cp0 normalizada por max_dist.
        expected = env._dist_to_finish() / env.params.max_dist
        assert obs[4] == pytest.approx(min(expected, 1.0))
    finally:
        os.environ.pop("LOCOMOTION_MODE", None)


# --- Integración: penalización off-track en el entorno ----------------------

def test_env_off_track_penalty_lowers_reward(fake_webots):
    import os
    os.environ["LOCOMOTION_MODE"] = "kinematic"
    try:
        supervisor = fake_webots.make_supervisor(self_name="ppo")
        env = DarwinRaceEnv(supervisor, robot_name="ppo")
        env.reset()
        # Tramo actual: cp0 (0,9) -> cp1 (8.79, 7.98). Un punto MUY lejos de esa
        # línea (y muy lejos de obstáculos) debe recibir penalización off-track.
        node = supervisor.getSelf()
        # Punto sobre la línea cerca del inicio (desviación ~0).
        node.set_translation([0.0, 9.0, 0.34])
        env.tracker.prev_dist = env._dist_to_target()
        _, reward_on, _, _, _ = env.step(1)  # girar, no avanza
        # Reubicar lejos de la línea de carrera (desviación grande), lejos de
        # vallas/robots para aislar la penalización off-track.
        env.reset()
        node.set_translation([0.0, -4.0, 0.34])
        env.tracker.prev_dist = env._dist_to_target()
        _, reward_off, _, _, _ = env.step(1)  # girar, no avanza
        assert reward_off < reward_on
    finally:
        os.environ.pop("LOCOMOTION_MODE", None)
