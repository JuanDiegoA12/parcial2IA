"""Controlador Webots del robot 'agente propio': agente propio (Taller 2).

Este es el AGENTE PROPIO del enunciado. A diferencia de PPO/DQN/A2C (que usan
stable-baselines3), aquí NO se importa stable-baselines3: el aprendizaje es una
implementación manual y elemental de Q-learning tabular sobre el mismo
DarwinRaceEnv. La idea es tener un algoritmo propio, legible y autocontenido,
que luego se pueda reemplazar por la solución final del Taller 2.

Lógica (Q-learning tabular):
    - La observación continua de 5 dimensiones se DISCRETIZA en bins, dando un
      estado entero indexable en una Q-table (numpy array).
    - Selección de acción epsilon-greedy (explora con probabilidad epsilon,
      explota argmax(Q[estado]) en caso contrario).
    - Actualización temporal-difference (TD) tras cada paso:
          Q[s,a] <- Q[s,a] + alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])
    - epsilon decae por episodio hacia un mínimo.

Dos modos (ver agent_runner.get_mode):
    train -> corre `episodes` episodios, aprende y PERSISTE la Q-table en
             models/custom_qtable.npz.
    race  -> carga la Q-table y actúa de forma greedy (epsilon=0).

======================================================================
PLACEHOLDER DEL TALLER 2
======================================================================
TODO(taller-2): reemplazar con el algoritmo del Taller 2.

Q-learning tabular es intencionalmente simple: es el punto de partida que se
debe sustituir por el algoritmo propio diseñado en el Taller 2 (p. ej. una
política basada en un método propio de búsqueda/control/aprendizaje).

Punto de extensión: para cambiar la estrategia de aprendizaje NO hace falta
reescribir todo el bucle. Basta con heredar de `TabularQLearningAgent` y
sobreescribir `select_action` y/o `update` (ver `CustomAgentExtensionPoint`
más abajo como esqueleto a completar). El resto del flujo (train/race,
persistencia, integración con los experimentos) se reutiliza tal cual.
======================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

# Permite importar el paquete común (agent_runner, race_env, config) tanto bajo
# el pythonpath de pytest como al ejecutarse como controlador de Webots.
_COMMON = os.path.join(os.path.dirname(__file__), "..", "darwin_common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import agent_runner  # noqa: E402  (tras ajustar sys.path)
import config as race_config  # noqa: E402

ROBOT_NAME = "agente propio"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "custom_qtable.npz"
)

# Episodios de entrenamiento por defecto (ajustable con EPISODES en el env).
TRAIN_EPISODES = int(os.environ.get("EPISODES", "300"))


class TabularQLearningAgent:
    """Agente de Q-learning tabular sobre un espacio de observación discretizado.

    Parameters
    ----------
    n_actions:
        Número de acciones discretas (len(config.ACTION_MAP) = 3).
    obs_low, obs_high:
        Cotas de la observación (del observation_space del entorno), usadas
        para mapear cada componente continuo a un bin.
    bins:
        Número de bins por dimensión de la observación.
    alpha, gamma:
        Tasa de aprendizaje y factor de descuento del TD-update.
    epsilon, epsilon_min, epsilon_decay:
        Parámetros de exploración epsilon-greedy (epsilon decae por episodio).
    seed:
        Semilla del generador aleatorio (reproducibilidad).
    """

    def __init__(
        self,
        n_actions: int,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        bins: int = 6,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99,
        seed: int = 0,
    ):
        self.n_actions = int(n_actions)
        self.obs_low = np.asarray(obs_low, dtype=np.float64)
        self.obs_high = np.asarray(obs_high, dtype=np.float64)
        self.n_dims = self.obs_low.shape[0]
        self.bins = int(bins)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)

        # Q-table densa: una fila por estado discreto, una columna por acción.
        # Estados = bins ** n_dims; con bins=6 y 5 dims son 7776 estados.
        self.q_table = np.zeros((self.bins ** self.n_dims, self.n_actions), dtype=np.float64)

    # --- discretización -----------------------------------------------------
    def discretize(self, obs: np.ndarray) -> int:
        """Mapea una observación continua a un índice de estado entero."""
        obs = np.asarray(obs, dtype=np.float64)
        span = self.obs_high - self.obs_low
        span = np.where(span == 0.0, 1.0, span)  # evita división por cero
        ratios = (obs - self.obs_low) / span
        idx = np.clip((ratios * self.bins).astype(int), 0, self.bins - 1)
        # Codificación posicional en base `bins` -> índice plano único.
        flat = 0
        for component in idx:
            flat = flat * self.bins + int(component)
        return flat

    # --- política -----------------------------------------------------------
    def select_action(self, state: int, greedy: bool = False) -> int:
        """Selecciona acción epsilon-greedy (greedy=True fuerza explotación).

        PUNTO DE EXTENSIÓN (Taller 2): sobreescribir este método para cambiar
        la estrategia de selección de acción sin tocar el bucle de carrera.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    # --- aprendizaje --------------------------------------------------------
    def update(self, state: int, action: int, reward: float, next_state: int,
               done: bool) -> None:
        """Actualización TD de Q-learning sobre la Q-table.

        PUNTO DE EXTENSIÓN (Taller 2): sobreescribir este método para cambiar
        la regla de aprendizaje (p. ej. otra función objetivo o traza elegible).
        """
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Reduce epsilon hacia epsilon_min tras cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # --- persistencia -------------------------------------------------------
    def save(self, path: str) -> None:
        """Guarda la Q-table y los metadatos de discretización en un .npz."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(
            path,
            q_table=self.q_table,
            bins=self.bins,
            obs_low=self.obs_low,
            obs_high=self.obs_high,
            n_actions=self.n_actions,
        )

    @classmethod
    def load(cls, path: str) -> "TabularQLearningAgent":
        """Recarga un agente desde un .npz previamente guardado."""
        # np.load añade '.npz' si falta; np.savez hace lo mismo al guardar.
        load_path = path if os.path.exists(path) else path + ".npz"
        data = np.load(load_path, allow_pickle=False)
        agent = cls(
            n_actions=int(data["n_actions"]),
            obs_low=data["obs_low"],
            obs_high=data["obs_high"],
            bins=int(data["bins"]),
        )
        agent.q_table = data["q_table"]
        agent.epsilon = 0.0  # en carga para carrera: explotación pura
        return agent

    # --- adaptador para agent_runner.run_race -------------------------------
    def predict(self, obs, deterministic: bool = True):
        """Interfaz tipo SB3 (model.predict) para reutilizar run_race.

        Devuelve (acción, estado) como SB3; el estado es None (sin memoria).
        """
        state = self.discretize(obs)
        action = self.select_action(state, greedy=deterministic)
        return action, None


class CustomAgentExtensionPoint(TabularQLearningAgent):
    """Esqueleto para el algoritmo final del Taller 2.

    TODO(taller-2): implementar aquí la estrategia propia sobreescribiendo
    `select_action` (cómo elegir la acción) y/o `update` (cómo aprender). El
    flujo de entrenamiento/carrera y la persistencia se heredan sin cambios.

    Ejemplo de uso una vez implementado:
        agent = CustomAgentExtensionPoint(n_actions, obs_low, obs_high)
        train_agent(env, agent=agent)
    """

    # Dejar intacto hasta tener el algoritmo del Taller 2; por defecto se
    # comporta como el Q-learning tabular base.
    pass


def build_agent(env, **kwargs) -> TabularQLearningAgent:
    """Construye un agente tabular dimensionado para `env`."""
    low = np.asarray(env.observation_space.low, dtype=np.float64)
    high = np.asarray(env.observation_space.high, dtype=np.float64)
    n_actions = int(env.action_space.n)
    return TabularQLearningAgent(n_actions, low, high, **kwargs)


def train_agent(env, episodes: int = TRAIN_EPISODES,
                agent: Optional[TabularQLearningAgent] = None,
                save_path: Optional[str] = None) -> TabularQLearningAgent:
    """Entrena el agente tabular sobre `env` durante `episodes` episodios.

    Parameters
    ----------
    env:
        DarwinRaceEnv ya construido (ver agent_runner.make_env).
    episodes:
        Número de episodios de entrenamiento.
    agent:
        Agente a entrenar; si es None se crea uno dimensionado para `env`.
    save_path:
        Si se indica, persiste la Q-table al terminar.

    Returns
    -------
    El agente entrenado (su Q-table queda actualizada).
    """
    if agent is None:
        agent = build_agent(env)

    for _ep in range(int(episodes)):
        obs, _info = env.reset()
        state = agent.discretize(obs)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_state = agent.discretize(next_obs)
            agent.update(state, action, float(reward), next_state,
                         done=bool(terminated))
            state = next_state
        agent.decay_epsilon()

    if save_path is not None:
        agent.save(save_path)
    return agent


def _build_backend():
    """Obtiene el Supervisor de Webots (solo disponible dentro del simulador)."""
    from controller import Supervisor

    return Supervisor()


def main():
    mode = agent_runner.get_mode()
    backend = _build_backend()
    env = agent_runner.make_env(ROBOT_NAME, backend)

    if mode == "train":
        train_agent(env, episodes=TRAIN_EPISODES, save_path=MODEL_PATH)
        print(f"[custom] Q-table entrenada y guardada en {MODEL_PATH}")
    else:
        agent = TabularQLearningAgent.load(MODEL_PATH)
        metrics = agent_runner.run_race(agent, env)
        print("[custom] métricas de carrera:", metrics)


if __name__ == "__main__":
    main()
