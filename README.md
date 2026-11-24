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
  locomotion_test/locomotion_test.py  # prueba de locomoción sin RL (LOCO_CMD)
experiments/
  run_experiments.py    # >=20 configuraciones -> results/experiments.csv
  plot_results.py        # tabla resumen + figuras comparativas (PNG)
models/                 # modelos entrenados (.zip SB3, .npz del agente propio)
results/                # CSV y figuras generadas
tests/                  # suite pytest con mock de Webots (no requiere simulador)
pista_atletismo_50m.wbt # mundo de Webots
```

## Locomoción (gait sinusoidal)

El DARwIn-OP **no** usa los managers de Robotis (`RobotisOp2GaitManager` /
`RobotisOp2MotionManager`): esos existen únicamente como binario **C++**, de
modo que `import managers` lanza `ModuleNotFoundError` en Python y no hay vía
oficial de Python para caminar. Por eso la locomoción se implementa accionando
**directamente los 20 motores** del robot con un **gait sinusoidal propio**
(`_GaitEngine` en `controllers/darwin_common/darwin_robot.py`), que ejecuta
`setPosition` sobre cada motor en **cada paso de simulación**.

El agente sigue eligiendo solo acciones discretas de alto nivel
(`0` walk_forward, `1` turn_left, `2` turn_right); el motor de gait las traduce
a una amplitud de avance (x) y una de giro (a), análogas a
`setXAmplitude`/`setAAmplitude` del controlador C++. Cada ángulo comandado se
recorta a `[min + margen, max - margen]`, así que **nunca** se pide a una
articulación su tope físico.

### Sobre las advertencias `RotationalMotor "LegLowerR/L": too big/low ... 2.25`

Provienen del **colapso** del robot cuando nada acciona sus motores (el stub
anterior solo llamaba a `supervisor.queue_action(...)`, un método inexistente
en Webots real), por lo que las piernas quedaban apoyadas contra el límite de la
rodilla (`[-2.25, 2.25]` rad). **No** las causa el código comandando ±2.25. Con
el gait sosteniendo las piernas cada paso (y el recorte con margen), las
articulaciones dejan de reposar contra sus topes y las advertencias desaparecen.
Esto se confirma empíricamente con el controlador `locomotion_test` (ver abajo).

### Parámetros ajustables (en `controllers/darwin_common/config.py`)

Todas las constantes del gait están centralizadas y comentadas; afínalas en
Webots observando el comportamiento real:

| Constante | Qué controla |
|-----------|--------------|
| `STANDING_POSE` | Ángulos nominales de la pose de pie (brazos/caderas/rodillas/tobillos). |
| `GAIT_PERIOD` / `GAIT_FREQUENCY` | Duración de un ciclo de marcha en pasos de simulación (uno deriva del otro). |
| `GAIT_X_AMPLITUDE` | Amplitud de avance al caminar hacia adelante. |
| `GAIT_A_AMPLITUDE` | Amplitud de giro al girar. |
| `GAIT_HIP_AMPLITUDE`, `GAIT_KNEE_AMPLITUDE`, `GAIT_ANKLE_AMPLITUDE` | Amplitud articular de cadera, rodilla y tobillo. |
| `GAIT_LATERAL_SWAY` | Balanceo lateral del peso (pelvis/tobillos). |
| `GAIT_LEG_PHASE_OFFSET` | Desfase entre piernas (180° = `math.pi`). |
| `MOTOR_VELOCITY` | Velocidad por defecto de los motores. |
| `JOINT_LIMITS`, `JOINT_LIMIT_MARGIN` | Límites por articulación y margen de seguridad. |
| `STABILIZATION_STEPS` | Ticks quietos de pie antes de cada episodio. |
| `FALL_TILT_THRESHOLD`, `ACCEL_CENTER`, `ACCEL_FALL_TOLERANCE`, `FALL_HEIGHT_Z` | Umbrales de detección de caída. |

> **Nota (asunción):** los límites articulares distintos de la rodilla usan
> rangos conservadores documentados en `config.py`; el PROTO `Darwin-op.proto`
> no pudo consultarse en el entorno de desarrollo. Ajústalos si observas recorte
> excesivo contra los topes reales del modelo.

### Detección de caída

`DarwinRobot.is_fallen()` combina tres señales (cualquiera activa = caído):
acelerómetro centrado en ~`ACCEL_CENTER` (512) con desviación mayor que
`ACCEL_FALL_TOLERANCE`, inclinación del torso (campo `rotation` del Supervisor)
sobre `FALL_TILT_THRESHOLD`, y altura `z` del torso bajo `FALL_HEIGHT_Z`.

### Validar la locomoción antes de entrenar (`locomotion_test`)

Primero valida **un solo robot** (que se mantenga de pie, avance y gire) con el
controlador de prueba, **sin** ejecutar ningún algoritmo de RL. Asigna el
controlador `locomotion_test` a un robot (campo `controller "locomotion_test"`)
y elige el comando con la variable de entorno `LOCO_CMD`:

```bash
LOCO_CMD=stand   webots pista_atletismo_50m.wbt   # mantener la pose de pie
LOCO_CMD=forward webots pista_atletismo_50m.wbt   # caminar hacia adelante
LOCO_CMD=left    webots pista_atletismo_50m.wbt   # girar a la izquierda
LOCO_CMD=right   webots pista_atletismo_50m.wbt   # girar a la derecha
```

(Opcional: `LOCO_ROBOT` elige el robot por nombre, por defecto `DQN`.) El
controlador registra en consola posición, heading, altura y acelerómetro cada
pocos pasos. **La validación física real de la caminata ocurre en Webots por
parte del usuario**: el entorno de desarrollo no tiene runtime de Webots.

Flujo recomendado: **(1)** validar la locomoción de un robot con
`locomotion_test`; **(2)** solo entonces integrar y entrenar los cuatro agentes.

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

> **Fallo conocido:** `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`
> falla porque el usuario reposicionó manualmente los robots dentro de Webots, de
> modo que las coordenadas del `.wbt` ya no coinciden con las esperadas. No lo
> provoca la locomoción y corregirlo exigiría editar la geometría del robot (lo
> cual está fuera de alcance). El resto de la suite pasa.

## Safe learning

El entorno (`race_env.py`) penaliza acercarse a obstáculos u otros robots dentro
de un radio de seguridad y termina el episodio ante colisiones duras o caídas,
de modo que los agentes aprenden a no dañarse ni dañar a otros.
