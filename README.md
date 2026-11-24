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

## Locomoción (dos planes: gait físico vs arrastre cinemático)

El proyecto se centra en la **carrera** y en **evaluar los agentes de RL**
(PPO, DQN, A2C y agente propio), **no** en caminar. Por eso hay **dos modos de
locomoción** intercambiables con una sola bandera, y ambos conservan las mismas
acciones discretas, el mismo `DarwinRaceEnv` y los mismos agentes:

- **Plan A — `gait` (por defecto):** gait sinusoidal físico que acciona los 20
  motores del DARwIn-OP cada paso (el `_GaitEngine` descrito abajo). Es la
  locomoción "de verdad" que exige la tarea (locomoción por motores, sin
  desplazamiento artificial vía `translation`) y la que deja **activa la
  detección de caída**. Es el modo por defecto.
- **Plan B — `kinematic` (OPT-IN):** arrastre cinemático tipo "tortuga"
  diferencial. El **Supervisor desliza al robot DE PIE** escribiendo
  directamente los campos `translation`/`rotation` cada paso (teletransporte por
  paso). **No** es el modo por defecto: sólo se activa explícitamente con
  `LOCOMOTION_MODE=kinematic`. Es *time-safe* (no hay que ajustar la marcha
  física), pero en este modo la caída está desactivada, por lo que queda como
  alternativa para pruebas rápidas, no como locomoción de entrega.

**Seleccionar el modo.** Vía la constante `config.LOCOMOTION_MODE`
(`"gait"` | `"kinematic"`, por defecto `"gait"`) o, con prioridad, la
variable de entorno `LOCOMOTION_MODE`. El modo se resuelve con
`config.resolve_locomotion_mode()` al construir el robot. Ejemplo:

```bash
LOCOMOTION_MODE=gait  webots pista_atletismo_50m.wbt   # Plan A (gait físico)
LOCOMOTION_MODE=kinematic webots pista_atletismo_50m.wbt  # Plan B (arrastre)
```

**Tunables del modo kinematic** (en `config.py`):

| Constante | Qué controla |
|-----------|--------------|
| `KINEMATIC_LINEAR_STEP` | Metros que avanza por tick de `walk_forward` (a lo largo del heading). |
| `KINEMATIC_ANGULAR_STEP` | Radianes que rota por tick de `turn_left`/`turn_right`. |

En Plan B el robot avanza por `KINEMATIC_LINEAR_STEP` a lo largo de su heading
(heading 0 mira a +y; avance `= (-sin θ·d, +cos θ·d, 0)`) y gira `±
KINEMATIC_ANGULAR_STEP` en z; cada paso se comanda la pose de pie a los motores
y se reinicia la física del nodo (`resetPhysics`) para que no acumule caída ni
deriva. En este modo la **caída está DESACTIVADA** (`is_fallen()` devuelve
siempre `False`), pero las **penalizaciones por proximidad/colisión** y el
**cruce de `GOAL_Y`** (victoria) siguen contando exactamente igual que en Plan A.

Para probar un robot aislado en modo arrastre (opt-in), antepón
`LOCOMOTION_MODE=kinematic` al lanzar `locomotion_test` (ver más abajo) y observa
en la consola cómo cambian posición y heading.

---

### Plan A — gait sinusoidal (detalle)

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

### Sobre las advertencias `RotationalMotor "...": too big/low requested position`

Estas advertencias provienen del **modelo Darwin-op REAL de Webots** (su PROTO
oficial con rangos de motor asimétricos y espejados), **no** del mock de los
tests (el mock nunca las emite). Hubo dos causas distintas:

1. `LegLowerR/L too big/low 2.25`: con el stub anterior nada accionaba los
   motores en Webots (su único efecto era `supervisor.queue_action(...)`,
   método inexistente en Webots real), el robot se desplomaba y la rodilla
   quedaba apoyada contra su tope de **un solo lado** (`2.25` en R, `-2.25` en
   L). No la causaba el código comandando ±2.25.
2. `LegUpperR 0.5 > 0.45`: este era un **bug nuestro**. `STANDING_POSE` pedía
   `0.5` cuando el tope real de `LegUpperR` es `0.45`, y el `JOINT_LIMITS`
   simétrico `(-1.57, 1.57)` de antes no lo recortaba.

Correcciones aplicadas: la pose de pie ahora queda dentro de los límites reales
con margen; los límites reales se **leen en runtime**
(`getMinPosition()`/`getMaxPosition()`) y se usan para el recorte final; y el
gait sostiene las piernas cada paso, así dejan de reposar contra sus topes.
Verifícalo en Webots con `locomotion_test` (modo diagnóstico, ver abajo).

Los límites reales (R2025a) son **asimétricos y espejados** entre izquierda y
derecha, p.ej. `LegUpperR [-1.77, 0.45]` vs `LegUpperL [-0.5, 1.68]`,
`LegLowerR [-0.02, 2.25]` vs `LegLowerL [-2.25, 0.03]`. Por eso la pose de pie
usa valores de signo invertido en cada lado. El PROTO es remoto; si Webots
imprime otros topes en el volcado diagnóstico, confía en los de runtime.

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
| `JOINT_LIMITS`, `JOINT_LIMIT_MARGIN` | Límites por articulación (FALLBACK) y margen de seguridad. |
| `STABILIZATION_STEPS` | Ticks quietos de pie antes de cada episodio (ahora `80`). |
| `FALL_HEIGHT_Z` | Altura `z` del torso bajo la cual = caída (señal primaria, `0.14`). |
| `FALL_TILT_THRESHOLD` | Inclinación del torso (ignorando yaw puro) que cuenta como vuelco. |
| `ACCEL_CENTER`, `ACCEL_FALL_TOLERANCE` | Centro (~512) y banda tolerada del acelerómetro. |
| `FALL_ACCEL_PERSIST_STEPS` | Ticks consecutivos fuera de banda antes de declarar caída por acelerómetro (`100`). |

> **Límites articulares:** `config.JOINT_LIMITS` es un **FALLBACK**. La fuente
> autoritativa son los límites **reales leídos en runtime** con
> `getMinPosition()`/`getMaxPosition()`, que `locomotion_test` vuelca por consola
> al arrancar (modo diagnóstico). Los valores del fallback provienen del PROTO
> oficial `Darwin-op.proto` de Webots R2025a y son asimétricos/espejados. Si el
> volcado diagnóstico muestra otros topes, cópialos a `JOINT_LIMITS`. *(Contenido
> reformulado por cumplimiento de licencia.)*

> **Pose de pie (`STANDING_POSE`):** el robot arranca en una **cuclilla modesta**
> (flexión suave de cadera/rodilla/tobillo), no totalmente erguido — el usuario
> confirmó que sólo se sostiene flexionado. Los valores son **puntos de partida
> para ajustar en Webots**: ejecuta `LOCO_CMD=stand`, observa la `z` resuelta
> (~0.23–0.25) y el equilibrio, y retoca cadera/rodilla/tobillo (recuerda el
> signo espejado L/R). Ajusta las `GAIT_*_AMPLITUDE`, `GAIT_PERIOD` y
> `MOTOR_VELOCITY` para la marcha/giro, y `STABILIZATION_STEPS` si no se asienta.

### Detección de caída

`DarwinRobot.is_fallen()` combina tres señales:

- **Altura (primaria):** `z` del torso bajo `FALL_HEIGHT_Z` (`0.14`) = caída
  inmediata. De pie/en cuclillas ronda `z ~ 0.23–0.25`; colapsado cae a ~`0.075`.
- **Inclinación:** el ángulo del torso (campo `rotation` del Supervisor) sobre
  `FALL_TILT_THRESHOLD`, **ignorando el yaw puro** (giro en z no es caída); sólo
  cuenta la inclinación horizontal.
- **Acelerómetro (con persistencia):** cada eje lee ~`ACCEL_CENTER` (512) de
  pie. La lectura es **ruidosa**, así que una desviación puntual mayor que
  `ACCEL_FALL_TOLERANCE` **no basta**: hace falta persistencia de
  `FALL_ACCEL_PERSIST_STEPS` (`100`) ticks consecutivos fuera de banda (al estilo
  del `Walk.cpp` oficial). Un pico aislado no marca `fallen=True`.

### Validar la locomoción antes de entrenar (`locomotion_test`)

Primero valida **un solo robot** (que se mantenga de pie, avance y gire) con el
controlador de prueba, **sin** ejecutar ningún algoritmo de RL.

> **Importante — dejar UN solo robot activo.** El usuario añadió un robot extra
> llamado **`arodillado`** con el controlador `locomotion_test`, de modo que hoy
> **dos** robots comparten ese controlador (`agente propio` y `arodillado`).
> Para la validación aislada, en el `.wbt` deja **solo uno** con
> `controller "locomotion_test"` (pon los demás en `"<none>"` o su controlador
> de agente) y revierte la simulación tras cada cambio.

> **Requisito — el robot debe ser Supervisor.** `locomotion_test` y el modo
> kinemático leen/escriben el nodo propio con `getSelf()`, que **sólo** funciona
> si el nodo del robot tiene `supervisor TRUE` en el `.wbt`. Si falta, Webots
> registra `illegal call to wb_supervisor_node_get_self() in a 'Robot'
> controller`, `getSelf()` devuelve `None` y antes el controlador reventaba con
> `AttributeError: 'NoneType' ... getSFVec3f`. Ahora `locomotion_test` detecta
> el caso, imprime un error accionable y sale limpiamente (`exit 1`) en vez de
> reventar. **Solución:** añade `supervisor TRUE` al nodo del robot al que
> asignes `locomotion_test`. (El `.wbt` lo gestionas tú; este repo no lo toca.)

El comando se resuelve con **prioridad**: `controllerArgs` (argv[1]) → variable
`LOCO_CMD` → por defecto `stand`. Comandos válidos: `stand | forward | left |
right`. En el `.wbt`, sobre el robot a validar:

```
controllerArgs ["stand"]     # mantener la pose de pie
controllerArgs ["forward"]   # caminar hacia adelante
controllerArgs ["left"]      # girar a la izquierda
controllerArgs ["right"]     # girar a la derecha
```

Edita `controllerArgs`, revierte (`Ctrl+Shift+T`) y vuelve a ejecutar en cada
cambio. Alternativa por entorno:

```bash
LOCO_CMD=forward webots pista_atletismo_50m.wbt
```

(Opcional: `LOCO_ROBOT` elige el robot por nombre, por defecto `DQN`.)

**Probar el modo kinematic (Plan B, opt-in) en un robot aislado.** Antepón
explícitamente `LOCOMOTION_MODE=kinematic` (no es el modo por defecto) para
arrastrar al robot de pie y ver cómo cambian posición y heading en la consola:

```bash
LOCOMOTION_MODE=kinematic LOCO_CMD=forward webots pista_atletismo_50m.wbt
```

`locomotion_test` imprime al arrancar el modo resuelto (`modo=gait|kinematic`).
En kinematic no hay que ajustar la marcha física y `is_fallen()` siempre es
`False`; el robot avanza/gira por `KINEMATIC_LINEAR_STEP`/`KINEMATIC_ANGULAR_STEP`.

**Modo diagnóstico:** al arrancar, el controlador imprime los **límites reales
(min/max) de los 20 motores** (una línea por motor) y, tras la estabilización,
la **`z` de pie resuelta**. Usa esos límites para fijar `JOINT_LIMITS` si
difieren, y la `z` para calibrar `STANDING_POSE` y `FALL_HEIGHT_Z`. El
controlador también registra posición/heading/altura/acelerómetro cada pocos
pasos. **La validación física real ocurre en Webots por parte del usuario**: el
entorno de desarrollo no tiene runtime de Webots.

Flujo recomendado: **(1)** con `locomotion_test` confirmar que un robot se
mantiene de pie (`stand`), avanza (`forward`) y gira (`left`/`right`) sin
advertencias `too big/low requested position`; **(2)** solo entonces integrar y
entrenar los cuatro agentes.

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
