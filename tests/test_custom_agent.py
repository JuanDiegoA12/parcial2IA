"""Tests del agente propio (Q-learning tabular) de controllers/custom_agent.

Cubren:
- que NO importa stable-baselines3,
- que la Q-table cambia tras entrenar unos episodios sobre el entorno mock,
- discretización y persistencia (save/load) de la Q-table,
- que run_race del agent_runner funciona con el agente (predict tipo SB3).

Todo corre con numpy+gymnasium sobre el backend mock (fixture fake_webots).
"""

import numpy as np

import agent_runner
import custom_agent


def _make_env(fake_webots):
    backend = fake_webots.make_supervisor(self_name="agente propio")
    return agent_runner.make_env("agente propio", backend)


def test_no_stable_baselines3_import():
    # El agente propio no debe IMPORTAR SB3 (las menciones en comentarios no
    # cuentan): ni en el código del módulo ni arrastrado a sys.modules.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(custom_agent))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "stable_baselines3" not in imported


def test_qtable_changes_after_training(fake_webots):
    env = _make_env(fake_webots)
    agent = custom_agent.build_agent(env, seed=0)
    before = agent.q_table.copy()

    custom_agent.train_agent(env, episodes=5, agent=agent)

    # La Q-table se modificó durante el aprendizaje.
    assert not np.array_equal(before, agent.q_table)
    assert np.any(agent.q_table != 0.0)


def test_discretize_bounds(fake_webots):
    env = _make_env(fake_webots)
    agent = custom_agent.build_agent(env, bins=6, seed=0)

    low = env.observation_space.low
    high = env.observation_space.high
    # Todo estado cae en el rango válido de la Q-table.
    for obs in (low, high, (low + high) / 2.0):
        state = agent.discretize(np.asarray(obs, dtype=np.float32))
        assert 0 <= state < agent.q_table.shape[0]


def test_save_and_load_roundtrip(tmp_path, fake_webots):
    env = _make_env(fake_webots)
    agent = custom_agent.build_agent(env, seed=0)
    custom_agent.train_agent(env, episodes=3, agent=agent)

    path = str(tmp_path / "qtable.npz")
    agent.save(path)
    reloaded = custom_agent.TabularQLearningAgent.load(path)

    assert np.array_equal(agent.q_table, reloaded.q_table)
    assert reloaded.bins == agent.bins
    assert reloaded.n_actions == agent.n_actions
    # En carga para carrera actúa greedy (epsilon=0).
    assert reloaded.epsilon == 0.0


def test_run_race_returns_metrics(fake_webots):
    env = _make_env(fake_webots)
    agent = custom_agent.build_agent(env, seed=0)
    custom_agent.train_agent(env, episodes=3, agent=agent)

    metrics = agent_runner.run_race(agent, env)
    assert set(metrics) == {
        "success", "steps", "cumulative_reward", "falls", "collisions",
        "checkpoints_reached",
    }
    assert isinstance(metrics["success"], bool)
    assert isinstance(metrics["steps"], int) and metrics["steps"] >= 1


def test_extension_point_overridable(fake_webots):
    # El punto de extensión del Taller 2 se puede subclasear y sigue funcionando.
    class AlwaysForward(custom_agent.CustomAgentExtensionPoint):
        def select_action(self, state, greedy=False):
            return 0  # siempre walk_forward

    env = _make_env(fake_webots)
    agent = AlwaysForward(
        n_actions=int(env.action_space.n),
        obs_low=env.observation_space.low,
        obs_high=env.observation_space.high,
    )
    custom_agent.train_agent(env, episodes=2, agent=agent)
    action, _state = agent.predict(env.observation_space.low)
    assert action == 0
