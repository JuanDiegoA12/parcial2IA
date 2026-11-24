# Carrera DARwIn-OP con aprendizaje por refuerzo (Webots)

Entorno para entrenar y comparar cuatro agentes que compiten en una carrera de
robots humanoides DARwIn-OP sobre una pista con obstáculos en Webots R2025a.

La pista es **OVALADA** y el punto de salida coincide con el de llegada (los
agentes arrancan en `y ≈ 9`, que es también la meta `y = GOAL_Y`). Con el
esquema antiguo "progreso hacia la meta fija `y = 9`" la recompensa estaba casi
asegurada sin dar la vuelta. Para corregirlo, ganar exige completar una
**VUELTA de CHECKPOINTS ORDENADOS**: `cp0 → cp1 → cp2 → … → cp7 → cp0` (una sola
vuelta). El robot debe alcanzar `cp1..cp7` en orden y recién entonces volver a
`cp0` (la meta). **Cruzar `y = 9` sin completar la secuencia NO gana.**

Los checkpoints `cp1..cp7` son nodos `Ball` del `.wbt` nombrados
`checkpoint1..checkpoint7`; se leen **por nombre** en runtime
(`DarwinRobot.find_checkpoints()`), que es la fuente autoritativa de sus
coordenadas. `cp0` (START/FINISH) no es un objeto nombrado: se **sintetiza** en
`(0, GOAL_Y)`. El entorno de entrenamiento y el detector de ganador comparten la
MISMA máquina de estados de la vuelta (`CheckpointTracker`), así que recompensa y
victoria coinciden por construcción.

> **Reentrenar los modelos:** la observación cambió de 5 a 7 dimensiones (ver
> abajo), por lo que los modelos entrenados con el esquema anterior **no son
> compatibles** y deben volver a entrenarse.

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
    race_env.py         # entorno gymnasium (observación 7-dim / acción / recompensa por checkpoints + CheckpointTracker)
    agent_runner.py     # helpers train/race para los agentes SB3
    race.py             # detector de ganador (primero en completar la vuelta de checkpoints)
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

- **Plan B — `kinematic` (por defecto):** arrastre cinemático tipo "tortuga"
  diferencial. El **Supervisor desliza al robot DE PIE** escribiendo
  directamente los campos `translation`/`rotation` cada paso (teletransporte por
  paso). Es una **decisión de diseño** del usuario: el foco es la **carrera** y
  **evaluar agentes**, no la marcha, así que este modo es *time-safe* (no hay
  que ajustar el gait físico). En este modo la caída está desactivada, pero el
  movimiento ahora es **realista** y los **obstáculos bloquean el avance** (ver
  abajo). Es el modo por defecto.
- **Plan A — `gait` (OPT-IN, sin cambios):** gait sinusoidal físico que acciona
  los 20 motores del DARwIn-OP cada paso (el `_GaitEngine` descrito abajo). Es la
  locomoción "de verdad" (por motores, sin desplazamiento artificial) y la que
  deja **activa la detección de caída**. Se activa con `LOCOMOTION_MODE=gait` y
  queda **exactamente igual que antes** (byte-for-byte).

**Seleccionar el modo.** Vía la constante `config.LOCOMOTION_MODE`
(`"gait"` | `"kinematic"`, por defecto `"kinematic"`) o, con prioridad, la
variable de entorno `LOCOMOTION_MODE`. El modo se resuelve con
`config.resolve_locomotion_mode()` al construir el robot. Ejemplo:

```bash
webots pista_atletismo_50m.wbt                            # Plan B (kinematic, por defecto)
LOCOMOTION_MODE=gait  webots pista_atletismo_50m.wbt      # Plan A (gait físico, opt-in)
```

#### Modo kinematic refinado (velocidad realista + colisiones + pin de pie)

**Velocidad realista.** `KINEMATIC_LINEAR_STEP = 0.016` m/tick. A
`basicTimeStep = 16 ms` (0.016 s/tick) eso da **1.0 m/s**, un paso humano
(antes 0.25 m/tick ≈ 15.6 m/s, irreal). El giro
`KINEMATIC_ANGULAR_STEP = 0.04` rad/tick ≈ **143 °/s**. La velocidad se afina en
**UN solo lugar** (`config.py`): tanto el desplazamiento real en Webots como el
mock de los tests leen estas mismas constantes.

**Los obstáculos BLOQUEAN el avance.** Antes de comprometer un `walk_forward`,
el modo kinematic calcula la `(x,y)` de destino; si cae dentro de
`KINEMATIC_COLLISION_RADIUS` (`0.3` m = semiancho del robot + semiespesor de la
valla) de alguna valla (`find_barriers`) u otro robot (`find_others`), el avance
se **cancela** para ese tick, de modo que el robot **no penetra** el obstáculo.
El **giro nunca se bloquea**, así el agente puede reorientarse y esquivar. El
radio queda por DEBAJO de `SAFETY_RADIUS` (0.6) para que la **penalización por
proximidad siga disparándose antes** del bloqueo, y por ENCIMA de
`HARD_COLLISION_DISTANCE` (0.15). Las penalizaciones por proximidad/colisión y
la condición de victoria (completar la **vuelta de checkpoints**) siguen
**activas** en `race_env`.

**Postura ERGUIDA de pie por tick (aproximación visual).** Cada tick, para
**TODAS** las acciones (`stand`/`walk_forward`/`turn_left`/`turn_right`) —no sólo
al avanzar— el modo kinematic endereza el nodo propio (`_apply_upright_pose` en
`darwin_robot.py`): reescribe la rotación como **yaw puro** sobre z (sin
roll/pitch, conservando el heading) y fija la altura del torso a
`KINEMATIC_STAND_Z` (`0.24`, z nominal de pie), de modo que el robot se deslice
**DE PIE** y no acostado/levitando. Además comanda a los motores la pose erguida
`KINEMATIC_STANDING_POSE` (**piernas casi rectas**, brazos a los lados) en vez de
la cuclilla `STANDING_POSE` del gait, y llama a `resetPhysics()`. Esta pose es
una **aproximación visual** ajustable que **NO afecta el movimiento** (que es
cinemático). El **movimiento y las colisiones no cambian**: sigue siendo el
deslizamiento de pie del racer visual, mientras que el gait físico (Plan A) queda
como el modo realista-pero-inestable opcional. Como aquí no hay runtime de
Webots, el usuario debe **verificar en Webots** que (1) la velocidad es ~1 m/s
con `LOCO_CMD=forward`; (2) el robot se detiene al llegar a una valla y puede
girar para esquivarla; (3) el torso se mantiene DE PIE a `z ~ 0.24` sin
acostarse —si **flota** sube `KINEMATIC_STAND_Z`, si se **hunde** bájalo—;
(4) `is_fallen()` sigue `False`.

**Tunables del modo kinematic** (todos en `config.py`, fuente única):

| Constante | Qué controla |
|-----------|--------------|
| `KINEMATIC_LINEAR_STEP` | Metros que avanza por tick de `walk_forward` (0.016 → 1.0 m/s a 16 ms). |
| `KINEMATIC_ANGULAR_STEP` | Radianes que rota por tick de `turn_left`/`turn_right` (0.04 → ~143 °/s). |
| `KINEMATIC_COLLISION_RADIUS` | Radio (0.3 m) dentro del cual una valla/robot delante bloquea el avance. |
| `KINEMATIC_STAND_Z` | Altura (0.24 m) a la que se fija el torso cada tick (pin visual de pie). Si el torso **flota**, súbelo; si se **hunde**, bájalo (de pie ~0.24–0.34). |
| `KINEMATIC_STANDING_POSE` | Pose ERGUIDA (piernas casi rectas, brazos a los lados) que se comanda a los motores en kinematic, en vez de la cuclilla `STANDING_POSE` del gait. Aproximación visual ajustable; NO afecta el movimiento (cinemático). |

En Plan B el robot avanza por `KINEMATIC_LINEAR_STEP` a lo largo de su heading
(heading 0 mira a +y; avance `= (-sin θ·d, +cos θ·d)`, con `z` fija a
`KINEMATIC_STAND_Z`) y gira `± KINEMATIC_ANGULAR_STEP` en z. La **caída está
DESACTIVADA** (`is_fallen()` devuelve siempre `False`), pero las
**penalizaciones por proximidad/colisión** y la **victoria** (completar la vuelta
de checkpoints ordenados) siguen contando exactamente igual que en Plan A.

Para probar un robot aislado en modo arrastre (por defecto), lanza
`locomotion_test` (ver más abajo) y observa en la consola cómo cambian posición
y heading; para forzar el gait físico antepón `LOCOMOTION_MODE=gait`.

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

**Probar el modo kinematic (Plan B, por defecto) en un robot aislado.** Sin
variables extra, `locomotion_test` ya corre en kinematic (modo por defecto) y
arrastra al robot de pie; para forzar el gait físico (Plan A) antepón
`LOCOMOTION_MODE=gait`:

```bash
LOCO_CMD=forward webots pista_atletismo_50m.wbt                 # kinematic (por defecto)
LOCOMOTION_MODE=gait LOCO_CMD=forward webots pista_atletismo_50m.wbt  # gait físico (opt-in)
```

`locomotion_test` imprime al arrancar el modo resuelto (`modo=gait|kinematic`).
En kinematic no hay que ajustar la marcha física y `is_fallen()` siempre es
`False`; el robot avanza por `KINEMATIC_LINEAR_STEP` (~1 m/s a 16 ms), gira por
`KINEMATIC_ANGULAR_STEP`, y **se detiene** ante vallas/robots (bloqueo de
colisión, ver arriba), pudiendo girar para esquivarlos.

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
**primer robot en COMPLETAR la vuelta de checkpoints ordenados**
(`cp0 → cp1 → … → cp7 → cp0`), no el primero en cruzar `y = GOAL_Y`. La lógica de
detección está en `controllers/darwin_common/race.py` (`RaceWinnerDetector` /
`run_race_until_winner`), que reutiliza el mismo `CheckpointTracker` del entorno
de entrenamiento y es verificable sin Webots con el mock de los tests.

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

## Vuelta de checkpoints (pista ovalada)

La pista es **ovalada** y la salida coincide con la llegada, así que premiar el
"progreso hacia `y = 9`" dejaba la recompensa casi asegurada sin dar la vuelta.
La solución es una **vuelta de CHECKPOINTS ORDENADOS**: el robot debe alcanzar
`cp1..cp7` **en orden** y luego volver a `cp0` (la meta) para ganar. Cruzar
`y = 9` sin completar la secuencia **no** gana. Es **una sola vuelta**.

- **Checkpoints por nombre.** `cp1..cp7` son nodos `Ball` del `.wbt`
  (`checkpoint1..checkpoint7`), leídos en runtime por
  `DarwinRobot.find_checkpoints()` (fuente autoritativa de sus coordenadas).
  `cp0` (START/FINISH) no es un objeto nombrado: se **sintetiza** en
  `(0, GOAL_Y)`. `config.CHECKPOINT_COORDS` guarda las coordenadas conocidas
  **sólo como fallback** documentado para el mock/tests y para degradar con
  gracia (con `RuntimeWarning`) si falta algún checkpoint en el mundo.
- **Orden de la vuelta:** `config.CHECKPOINT_ORDER = (1, 2, 3, 4, 5, 6, 7, 0)`
  → `TOTAL_TARGETS = 8` objetivos a alcanzar en orden.
- **Sólo cuenta el SIGUIENTE checkpoint.** Alcanzar un checkpoint posterior
  antes de tiempo se ignora; sólo el objetivo actual avanza la vuelta.
- **Estado compartido.** El entorno y el detector de ganador usan la MISMA
  máquina de estados (`CheckpointTracker`), así que recompensa y victoria
  coinciden por construcción.

### Observación (7 dimensiones, antes 5)

| idx | componente | normalización |
|-----|------------|---------------|
| 0 | distancia al SIGUIENTE checkpoint | `min(dist / max_dist, 1)` ∈ [0, 1] |
| 1 | ángulo al siguiente checkpoint | `clip(ángulo / π, -1, 1)` |
| 2 | heading del robot | `clip(heading / π, -1, 1)` |
| 3 | progreso de la vuelta | `reached_count / TOTAL_TARGETS` ∈ [0, 1] |
| 4 | índice normalizado del siguiente checkpoint | `target_ordinal / TOTAL_TARGETS` ∈ [0, 1] |
| 5 | distancia al obstáculo más cercano | `1` si ninguno, si no `min(dist / max_dist, 1)` |
| 6 | distancia al robot más cercano | `1` si ninguno, si no `min(dist / max_dist, 1)` |

> **Reentrenar:** la observación cambió de 5 a 7 dimensiones, así que los
> modelos previos **no** son compatibles y deben volver a entrenarse. Los
> controladores leen la dimensión del `observation_space`, así que no hay nada
> hardcodeado que tocar; sólo hay que reentrenar.

### Recompensa por checkpoints

- **+ progreso continuo** hacia el siguiente checkpoint:
  `CHECKPOINT_PROGRESS_WEIGHT · (reducción de distancia al objetivo este paso)`.
- **+ bono discreto** `CHECKPOINT_BONUS` al **alcanzar** cada checkpoint
  intermedio (dentro de `CHECKPOINT_RADIUS`); luego avanza el objetivo.
- **+ `GOAL_BONUS`** y `terminated` al **cerrar la vuelta** (volver a `cp0` tras
  `cp1..cp7`).
- **− penalización off-track:** distancia del robot al **segmento de recta**
  entre el predecesor y el objetivo actual (la línea de carrera), más allá de
  `OFF_TRACK_TOLERANCE`, escalada por `OFF_TRACK_PENALTY_WEIGHT`. No termina el
  episodio; es determinista.
- Se mantienen **sin cambios** las penalizaciones de safe-learning: proximidad a
  obstáculos/otros robots, colisión dura (termina), caída (termina) y el pequeño
  costo por paso.

### Tunables de la vuelta (en `config.py`)

| Constante | Qué controla |
|-----------|--------------|
| `CHECKPOINT_RADIUS` | Radio (m, `1.0`) para dar por alcanzado un checkpoint; también el radio de victoria alrededor de `cp0`. |
| `CHECKPOINT_PROGRESS_WEIGHT` | Peso del progreso continuo hacia el siguiente checkpoint (`10.0`). |
| `CHECKPOINT_BONUS` | Bono discreto al alcanzar cada checkpoint intermedio (`20.0`). |
| `OFF_TRACK_PENALTY_WEIGHT` | Peso de la penalización por desviarse de la línea de carrera (`2.0`). |
| `OFF_TRACK_TOLERANCE` | Desviación (m, `1.0`, ~semiancho de carril) tolerada antes de penalizar. |
| `CHECKPOINT_ORDER` | Orden de la vuelta (`(1,2,3,4,5,6,7,0)`), `TOTAL_TARGETS = 8`. |
| `CHECKPOINT_COORDS` | Coordenadas fallback de `cp0..cp7` para el mock/tests (la fuente autoritativa es `find_checkpoints()`). |

Las métricas de los experimentos añaden una columna `checkpoints_reached`
(máximo de checkpoints alcanzados por episodio); las columnas anteriores se
conservan y `success` ahora significa **"completó la vuelta"**.
