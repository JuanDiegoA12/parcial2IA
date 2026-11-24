"""Comparación y graficado de los resultados de la carrera DARwIn-OP.

Lee results/experiments.csv, calcula promedios por algoritmo (tabla resumen
guardada en results/summary.csv e impresa en consola) y genera dos figuras
comparativas con matplotlib (backend 'Agg', sin ventana):

  - results/cumulative_reward.png : recompensa acumulada media por algoritmo.
  - results/success_rate.png      : tasa de éxito media por algoritmo.

Uso:
    python3 experiments/plot_results.py
    python3 experiments/plot_results.py --input results/experiments.csv

matplotlib y pandas se importan de forma perezosa dentro de las funciones.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "results")

EXPERIMENTS_CSV = os.path.join(RESULTS_DIR, "experiments.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")
CUMULATIVE_REWARD_PNG = os.path.join(RESULTS_DIR, "cumulative_reward.png")
SUCCESS_RATE_PNG = os.path.join(RESULTS_DIR, "success_rate.png")


def summarize(input_csv: str = EXPERIMENTS_CSV, summary_csv: str = SUMMARY_CSV):
    """Promedia las métricas por algoritmo y guarda/imprime la tabla resumen.

    Returns
    -------
    Un pandas.DataFrame indexado por 'algo' con las medias por columna.
    """
    import pandas as pd  # import perezoso

    df = pd.read_csv(input_csv)
    # Agregaciones base (compatibles con CSVs antiguos sin checkpoints_reached).
    agg = dict(
        cumulative_reward=("cumulative_reward", "mean"),
        success_rate=("success", "mean"),
        steps_to_goal=("steps_to_goal", "mean"),
        falls=("falls", "mean"),
        collisions=("collisions", "mean"),
        n_configs=("config_id", "count"),
    )
    # Columna nueva (vuelta de checkpoints): sólo si el CSV la trae, para seguir
    # cargando CSVs antiguos sin ella. success ahora = "completó la vuelta".
    if "checkpoints_reached" in df.columns:
        agg["checkpoints_reached"] = ("checkpoints_reached", "mean")
    summary = df.groupby("algo").agg(**agg)

    directory = os.path.dirname(summary_csv)
    if directory:
        os.makedirs(directory, exist_ok=True)
    summary.to_csv(summary_csv)
    print("Resumen por algoritmo (promedios):")
    print(summary.to_string())
    return summary


def _bar_figure(summary, column: str, title: str, ylabel: str, output_png: str) -> None:
    """Dibuja un gráfico de barras de `column` por algoritmo y lo guarda."""
    import matplotlib

    matplotlib.use("Agg")  # backend sin ventana (headless)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    algos = list(summary.index)
    values = list(summary[column])
    # Un color por algoritmo (PPO/DQN/A2C + agente propio CUSTOM).
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ax.bar(algos, values, color=palette[: len(algos)])
    ax.set_title(title)
    ax.set_xlabel("Algoritmo")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    directory = os.path.dirname(output_png)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_results(
    input_csv: str = EXPERIMENTS_CSV,
    summary_csv: str = SUMMARY_CSV,
    cumulative_reward_png: str = CUMULATIVE_REWARD_PNG,
    success_rate_png: str = SUCCESS_RATE_PNG,
):
    """Flujo completo: resumen + las dos figuras comparativas.

    Returns
    -------
    El DataFrame resumen (también escrito en summary_csv).
    """
    summary = summarize(input_csv=input_csv, summary_csv=summary_csv)
    _bar_figure(
        summary,
        column="cumulative_reward",
        title="Recompensa acumulada media por algoritmo",
        ylabel="Recompensa acumulada",
        output_png=cumulative_reward_png,
    )
    _bar_figure(
        summary,
        column="success_rate",
        title="Tasa de éxito media por algoritmo",
        ylabel="Tasa de éxito",
        output_png=success_rate_png,
    )
    print(f"Figuras escritas en {cumulative_reward_png} y {success_rate_png}")
    return summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grafica la comparación de resultados de la carrera."
    )
    parser.add_argument(
        "--input",
        default=EXPERIMENTS_CSV,
        help="CSV de entrada (por defecto results/experiments.csv).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    plot_results(input_csv=args.input)


if __name__ == "__main__":
    main()
