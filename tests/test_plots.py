"""Tests de experiments/plot_results.py.

Con un CSV sintético pequeño, plot_results debe escribir las dos figuras PNG y
la tabla resumen summary.csv. Todo se protege con importorskip('matplotlib')
(que arrastra pandas) para que se salte limpio si la lib no está instalada.
"""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

import pandas as pd

from experiments import plot_results as pr


def _write_synthetic_csv(path) -> None:
    rows = [
        {"algo": "PPO", "config_id": "PPO-000", "space_size": 60.0, "start": "default",
         "goal": 9.0, "cumulative_reward": 120.5, "success": True,
         "steps_to_goal": 40, "falls": 0, "collisions": 0},
        {"algo": "PPO", "config_id": "PPO-001", "space_size": 30.0, "start": "default",
         "goal": 8.0, "cumulative_reward": 90.0, "success": False,
         "steps_to_goal": -1, "falls": 1, "collisions": 0},
        {"algo": "DQN", "config_id": "DQN-000", "space_size": 60.0, "start": "default",
         "goal": 9.0, "cumulative_reward": 60.0, "success": True,
         "steps_to_goal": 55, "falls": 0, "collisions": 1},
        {"algo": "A2C", "config_id": "A2C-000", "space_size": 90.0, "start": "default",
         "goal": 10.0, "cumulative_reward": 75.0, "success": False,
         "steps_to_goal": -1, "falls": 0, "collisions": 0},
    ]
    pd.DataFrame(rows, columns=pr_columns()).to_csv(path, index=False)


def pr_columns():
    from experiments.run_experiments import CSV_COLUMNS

    return CSV_COLUMNS


def test_plot_results_writes_outputs(tmp_path):
    input_csv = tmp_path / "experiments.csv"
    _write_synthetic_csv(input_csv)

    summary_csv = tmp_path / "summary.csv"
    reward_png = tmp_path / "cumulative_reward.png"
    success_png = tmp_path / "success_rate.png"

    summary = pr.plot_results(
        input_csv=str(input_csv),
        summary_csv=str(summary_csv),
        cumulative_reward_png=str(reward_png),
        success_rate_png=str(success_png),
    )

    assert summary_csv.exists()
    assert reward_png.exists() and reward_png.stat().st_size > 0
    assert success_png.exists() and success_png.stat().st_size > 0

    # Resumen: una fila por algoritmo con la media correcta.
    assert set(summary.index) == {"PPO", "DQN", "A2C"}
    assert summary.loc["PPO", "cumulative_reward"] == pytest.approx((120.5 + 90.0) / 2)
    assert summary.loc["PPO", "success_rate"] == pytest.approx(0.5)

    # summary.csv recargable y con las columnas esperadas.
    reloaded = pd.read_csv(summary_csv)
    assert "algo" in reloaded.columns
    for col in ("cumulative_reward", "success_rate"):
        assert col in reloaded.columns
