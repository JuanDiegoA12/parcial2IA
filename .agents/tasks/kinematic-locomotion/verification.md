# Verificación — modo de locomoción cinemático (Plan B)

Todo se ejecutó con el venv del proyecto (`.venv/bin/python`); **NO** se lanzó
Webots (entorno sin runtime de simulador). El trabajo es **aditivo**: el gait
(Plan A) queda intacto; sólo se añade un segundo modo seleccionado por bandera.

## Qué se cambió (resumen)

- `controllers/darwin_common/config.py`: `import os`; bloque-comentario Plan A
  vs Plan B; constantes nuevas `LOCOMOTION_MODE="kinematic"` (default),
  `KINEMATIC_LINEAR_STEP=0.25`, `KINEMATIC_ANGULAR_STEP=0.20`; función
  `resolve_locomotion_mode()` (env `LOCOMOTION_MODE` con prioridad, validada a
  `{gait,kinematic}`, fallback al default). No se removió ni renumeró ninguna
  constante existente.
- `controllers/darwin_common/darwin_robot.py`:
  - `_LocomotionBackend.__init__` resuelve `self.mode` y añade `self._kin_action`.
  - `walk_forward/turn_left/turn_right/stand` conservan las líneas de gait
    (`set_amplitudes` + `_queue`) y, SÓLO en kinematic, registran `_kin_action`.
  - `_apply_kinematic_motion(action)`: escribe `translation`/`rotation` del nodo
    propio con la convención del mock, **sólo cuando el backend NO expone
    `queue_action`** (= Webots real), para no duplicar el desplazamiento bajo el
    mock (que ya lo produce vía `_apply_kinematics`).
  - `update()`: rama kinematic = `hold_standing()` + `_apply_kinematic_motion()`
    + `_reset_self_physics()` + limpiar `_kin_action`; rama gait EXACTAMENTE
    igual que antes (`engine.advance()` + `apply_targets(...)`).
  - `is_fallen()`: guarda temprana `if self.locomotion.mode == "kinematic":
    return False`.
  - `reset_pose()`: limpia `_kin_action` al reiniciar el episodio.
- `controllers/darwin_common/race_env.py`: **sin cambios** (ya es
  mode-agnostic; proximidad/colisión/meta intactos; `check_env` pasa en ambos
  modos sin tocar nada).
- `controllers/locomotion_test/locomotion_test.py`: imprime el modo resuelto
  (`modo=gait|kinematic`) y una nota en kinematic. Sin ramas de comportamiento
  (el ruteo por modo vive en `DarwinRobot`).
- Tests: nuevo `tests/test_kinematic_locomotion.py`; añadido
  `test_check_env_both_modes` y `test_obstacle_intrusion_penalizes_in_kinematic_mode`
  en `tests/test_race_env.py`. En `tests/test_darwin_robot.py` y
  `tests/test_race_env.py` los tests que verifican la DETECCIÓN DE CAÍDA / avance
  de fase del gait (comportamiento de Plan A, que kinematic desactiva por
  diseño) ahora fijan `LOCOMOTION_MODE="gait"` antes de construir el robot, con
  un fixture autouse que restaura la variable. Las aserciones no cambian.
- `README.md`: subsección "dos planes" (gait vs kinematic), cómo seleccionar el
  modo, los dos tunables, que el arrastre es por diseño, que la caída se
  desactiva pero colisiones/meta cuentan, y cómo probar un robot aislado.

## Decisión clave (anti doble-desplazamiento bajo el mock)

El mock (`tests/conftest.py`) expone `queue_action` y mueve el robot en
`step()` → `_apply_kinematics()` con `WALK_DELTA=0.25`, `TURN_DELTA=0.20` y la
convención `theta = rot[3] if rot[2]>=0 else -rot[3]`, avance
`(-sin·d, +cos·d)`. Por eso `KINEMATIC_LINEAR_STEP=0.25` y
`KINEMATIC_ANGULAR_STEP=0.20` (iguales a los deltas del mock) y el field-write
directo de `_apply_kinematic_motion` se **salta** cuando hay `queue_action`.
Resultado: bajo el mock mueve el mock; en Webots real mueve el field-write; cada
entorno tiene exactamente un "motor" y las magnitudes coinciden.

## Comandos ejecutados y resultados

### 1. Suite completa — modo por defecto (kinematic)

```
.venv/bin/python -m pytest -q
```
Resultado: **82 passed, 1 failed**. El único fallo es el pre-existente y fuera
de alcance `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged` (el
usuario movió los robots en el `.wbt`). El baseline antes de este trabajo era
65 passed / 1 failed; los 17 nuevos "passed" son los tests kinematic añadidos.

### 2. Suite completa — modo gait

```
LOCOMOTION_MODE=gait .venv/bin/python -m pytest -q
```
Resultado: **82 passed, 1 failed** (mismo `test_wbt` conocido). El
comportamiento del gait no cambia según el modo global porque los tests de
gait/caída fijan su propio modo; confirma que el ruteo por modo no rompe nada.

### 3. check_env de stable-baselines3 en AMBOS modos

```
.venv/bin/python -m pytest -v tests/test_race_env.py::test_check_env_both_modes
```
Resultado:
```
tests/test_race_env.py::test_check_env_both_modes[gait] PASSED
tests/test_race_env.py::test_check_env_both_modes[kinematic] PASSED
2 passed
```
(stable-baselines3 está instalado; el test NO se omitió.)

### 4. Tests unitarios del modo kinematic

```
.venv/bin/python -m pytest -q tests/test_kinematic_locomotion.py
```
Cubren: `walk_forward`+`tick` a heading 0 mueve +y por `KINEMATIC_LINEAR_STEP`
(x y z sin cambio); `turn_left`/`turn_right`+`tick` cambian heading por
`±KINEMATIC_ANGULAR_STEP`; avance tras giro sigue el nuevo heading; `is_fallen()`
= False en kinematic aun con tilt fuerte y z baja; contraprueba gait donde
`is_fallen()` = True; `reset_pose` limpia la acción pendiente. Todos pasan.

Penalización por proximidad a obstáculos en kinematic:
`tests/test_race_env.py::test_obstacle_intrusion_penalizes_in_kinematic_mode`
confirma que `margin_violations >= 1` y `reward < 0` con el modo kinematic
activo (sólo la caída se desactiva, las colisiones no).

### 5. Compilación byte-a-byte de las fuentes cambiadas

```
.venv/bin/python -m py_compile \
  controllers/darwin_common/config.py \
  controllers/darwin_common/darwin_robot.py \
  controllers/darwin_common/race_env.py \
  controllers/locomotion_test/locomotion_test.py
```
Resultado: exit 0 (COMPILE OK).

### 6. Sanidad de config

```
.venv/bin/python -c "import sys; sys.path.insert(0,'controllers/darwin_common'); \
  import config; print(config.LOCOMOTION_MODE, config.KINEMATIC_LINEAR_STEP, \
  config.KINEMATIC_ANGULAR_STEP, config.resolve_locomotion_mode())"
```
Imprime: `kinematic 0.25 0.2 kinematic`.

## Nota sobre concurrencia

Otro workflow edita en paralelo los mismos archivos para ajustar Plan A. Se
releyó cada archivo justo antes de editarlo y se construyó sobre el estado en
disco (que ya traía `_clamp_to_limits`, `motor_limits`, `FALL_ACCEL_PERSIST_STEPS`,
etc.). Todas las ediciones son aditivas: ramas nuevas guardadas por
`mode == "kinematic"`, constantes nuevas al final del bloque de constantes, sin
tocar el código ni las constantes del gait ni las firmas públicas.
```
```
