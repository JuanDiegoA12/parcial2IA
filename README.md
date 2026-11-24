# Carrera DARwIn-OP con aprendizaje por refuerzo (Webots)

Entorno para entrenar y comparar cuatro agentes que compiten en una carrera de
robots humanoides DARwIn-OP sobre una pista con obstáculos en Webots R2025a. El
robot que primero cruza la línea de meta (plano `y = GOAL_Y`, con `GOAL_Y = 9`)
gana la carrera.

## Agentes

| Robot (`name` en el .wbt) | Carpeta del controlador | Algoritmo |
|---------------------------|-------------------------|-----------|
| `DQN`                     | `controllers/dqn_agent`    | DQN (stable-baselines3) |
| `ppo`                     | `controllers/ppo_agent`    | PPO (stable-baselines3) |
| `actor - critic`          | `controllers/a2c_agent`    | A2C (stable-baselines3) |
| `agente propio`           | `controllers/custom_agent` | **Agente propio**: Q-learning tabular manual (placeholder del Taller 2) |

> El **agente propio** es intencionalmente elemental (Q-learning tabular, sin
> stable-baselines3) y está marcado como **placeholder del Taller 2**. Para
> sustituirlo por el algoritmo final, ver el punto de extensión
> `CustomAgentExtensionPoint` en `controllers/custom_agent/custom_agent.py`
> (basta sobreescribir `select_action` y/o `update`).

## Estructura del proyecto

```
controllers/
  darwin_common/        # capa compartida (reutilizada por los 4 agentes)
    config.py           # constantes del mundo, acciones, pesos de recompensa
    darwin_robot.py     # robot de alto nivel sobre el Supervisor de Webots
    race_env.py         # entorno gymnasium (observación/acción/recompensa)
    agent_runner.py     # helpers train/race para los agentes SB3
    race.py             # detector de ganador (primer robot que cruza la meta)
  ppo_agent/ppo_agent.py
  dqn_agent/dqn_agent.py
  a2c_agent/a2c_agent.py
  custom_agent/custom_agent.py   # agente propio (Taller 2)
experiments/
  run_experiments.py    # >=20 configuraciones -> results/experiments.csv
  plot_results.py        # tabla resumen + figuras comparativas (PNG)
models/                 # modelos entrenados (.zip SB3, .npz del agente propio)
results/                # CSV y figuras generadas
tests/                  # suite pytest con mock de Webots (no requiere simulador)
pista_atletismo_50m.wbt # mundo de Webots
```

## 1. Preparar el entorno de Python

Webots **ya está instalado** por el usuario y **no** lo instala ni lo lanza
ningún script de este repo: se abre manualmente desde la aplicación de Webots.

Lo único que hay que preparar es un entorno de Python con las librerías:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` incluye gymnasium, stable-baselines3, numpy, pandas,
matplotlib y tensorflow. El agente propio solo usa numpy.

Para que Webots use este intérprete, configúralo en
`Tools > Preferences > Python command` (apuntando al Python del entorno virtual)
o define la variable de entorno correspondiente antes de abrir Webots.

## 2. Entrenar cada agente (`MODE=train`)

Cada controlador tiene dos modos, seleccionados por la variable de entorno
`MODE` (o el primer argumento del controlador): `train` o `race`
(por defecto `race`).

Para **entrenar**, abre Webots con `pista_atletismo_50m.wbt`, pon el controlador
del robot en modo entrenamiento y ejecuta la simulación. El modo se lee de
`MODE`; puedes fijarlo en los argumentos del controlador (campo `controllerArgs`)
o como variable de entorno al lanzar Webots:

```bash
MODE=train webots pista_atletismo_50m.wbt
```

Al entrenar, cada agente guarda su modelo:

- `ppo_agent`  -> `models/ppo.zip`
- `dqn_agent`  -> `models/dqn.zip`
- `a2c_agent`  -> `models/a2c.zip`
- `custom_agent` (agente propio) -> `models/custom_qtable.npz`

Variables útiles: `TIMESTEPS` (pasos de entrenamiento SB3, por defecto 100000) y
`EPISODES` (episodios del agente propio, por defecto 300).

## 3. Experimentos (>=20 configuraciones)

Fuera de Webots, los experimentos entrenan cada agente contra un backend mock y
registran métricas (recompensa acumulada, tasa de éxito, etc.) variando tamaño
del espacio, posición inicial, posición de la meta e hiperparámetros:

```bash
python experiments/run_experiments.py            # set completo (>=20 configs)
python experiments/run_experiments.py --quick     # set reducido (CI)
```

Salida: `results/experiments.csv`.

## 4. Comparación y gráficos

```bash
python experiments/plot_results.py
```

Genera `results/summary.csv` (promedios por algoritmo) y las figuras
`results/cumulative_reward.png` y `results/success_rate.png`, comparando los
cuatro algoritmos (PPO, DQN, A2C y agente propio).

## 5. Carrera final (`MODE=race`)

Con los modelos ya entrenados, abre Webots y corre todos los controladores en
modo carrera para la competencia:

```bash
MODE=race webots pista_atletismo_50m.wbt
```

Cada controlador carga su modelo y actúa de forma determinista. El ganador es el
primer robot en cruzar `y = GOAL_Y`; la lógica de detección está en
`controllers/darwin_common/race.py` (`RaceWinnerDetector` /
`run_race_until_winner`) y es verificable sin Webots con el mock de los tests.

## Pruebas

La suite corre sin Webots (mock de la API en `tests/conftest.py`):

```bash
python3 -m pytest tests/ -q
```

Los tests que dependen de stable-baselines3, tensorflow o matplotlib se saltan
limpiamente si esas librerías no están instaladas.

## Safe learning

El entorno (`race_env.py`) penaliza acercarse a obstáculos u otros robots dentro
de un radio de seguridad y termina el episodio ante colisiones duras o caídas,
de modo que los agentes aprenden a no dañarse ni dañar a otros.
