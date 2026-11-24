"""Tests de experiments/run_experiments.py.

- El generador de configuraciones produce >=20 configuraciones distintas que
  varían las cuatro dimensiones requeridas (tamaño de espacio, inicio, meta,
  hiperparámetros). Esto NO requiere SB3 ni ejecuta nada.
- El modo --quick escribe el CSV con las columnas esperadas y >=1 fila. La
  ejecución depende de stable-baselines3, así que se protege con importorskip.
"""

from experiments import run_experiments as rx


def test_generate_configs_at_least_20():
    configs = rx.generate_configs()
    assert len(configs) >= 20
    # IDs únicos (configuraciones distintas).
    assert len({c.config_id for c in configs}) == len(configs)


def test_generate_configs_varies_all_four_dimensions():
    configs = rx.generate_configs()
    assert len({c.space_size for c in configs}) >= 2          # tamaño de espacio
    assert len({c.start for c in configs}) >= 2               # posición inicial
    assert len({c.goal for c in configs}) >= 2                # posición de meta
    # Hiperparámetros: al menos dos combinaciones distintas a lo largo del set.
    hp_variants = {tuple(sorted(c.hyperparams.items())) for c in configs}
    assert len(hp_variants) >= 2
    # Cubre los tres algoritmos SB3 (el propio llega en FEAT-004).
    assert {c.algo for c in configs} == {"PPO", "DQN", "A2C"}


def test_quick_configs_one_per_algo():
    quick = rx.generate_quick_configs()
    assert len(quick) == 3
    assert {c.algo for c in quick} == {"PPO", "DQN", "A2C"}


def test_csv_schema_constant():
    # Esquema estable consumido por plot_results y FEAT-004.
    assert rx.CSV_COLUMNS == [
        "algo",
        "config_id",
        "space_size",
        "start",
        "goal",
        "cumulative_reward",
        "success",
        "steps_to_goal",
        "falls",
        "collisions",
    ]


def test_quick_run_writes_csv(tmp_path):
    # La ejecución real necesita SB3 (y pandas); se salta si falta.
    import pytest

    pytest.importorskip("stable_baselines3")
    pytest.importorskip("pandas")
    import pandas as pd

    out = tmp_path / "experiments.csv"
    path = rx.run_experiments(quick=True, output_csv=str(out))
    assert path == str(out)
    assert out.exists()

    df = pd.read_csv(out)
    assert list(df.columns) == rx.CSV_COLUMNS
    assert len(df) >= 1
    # Columnas clave exigidas por los criterios de aceptación.
    for col in ("algo", "cumulative_reward", "success", "steps_to_goal",
                "falls", "collisions"):
        assert col in df.columns
