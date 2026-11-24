# Verificación — Postura ERGUIDA del modo cinemático (Plan B)

Objetivo: corregir SÓLO la POSTURA VISUAL del arrastre cinemático (el robot se
deslizaba acostado boca arriba / levitando). Todo lo demás del modo kinematic
(velocidad ~1 m/s, bloqueo de colisión, `is_fallen()=False`, z fija a 0.24) y el
modo gait (Plan A) quedan sin cambios de comportamiento.

No hay runtime de Webots: verificación con `pytest` + mock, `check_env` de
stable-baselines3 y `py_compile`. NO se lanzó Webots.

## Cambios implementados

- `controllers/darwin_common/config.py`:
  - Nueva constante `KINEMATIC_STANDING_POSE` (20 motores, pose erguida: piernas
    casi rectas con rodilla levemente flexionada R+0.1/L-0.1, caderas/tobillos
    neutros, brazos a los lados, cuello/cabeza neutros). Todos los ángulos dentro
    de `JOINT_LIMITS` con holgura ≥ `JOINT_LIMIT_MARGIN` (0.05). Comentario en
    español: aproximación visual ajustable que NO afecta el movimiento.
  - Comentario de `KINEMATIC_STAND_Z` ampliado con guía de ajuste (sube si flota,
    baja si se hunde; de pie ~0.24–0.34). Valor sin cambios: `0.24`.
- `controllers/darwin_common/darwin_robot.py`:
  - Nuevo método `_LocomotionBackend._apply_upright_pose()`: escribe la postura
    erguida del nodo propio CADA tick para TODAS las acciones, SIN la guarda
    `queue_action` (observable bajo el mock). Rotación = yaw puro `[0,0,1,theta]`
    con la misma convención de heading; `translation` conserva x,y y fija
    `z = KINEMATIC_STAND_Z`. Defensivo con try/except.
  - `_apply_kinematic_motion` rama `walk_forward`: ahora escribe SÓLO x,y
    (conserva la z vigente); el pin de z lo hace `_apply_upright_pose` el mismo
    tick. Ramas `turn_left`/`turn_right` y la fórmula de avance intactas.
  - `update()` modo kinematic: comanda `hold_kinematic_standing()` (pose erguida
    recortada con `clamp`) en vez de `hold_standing()` (cuclilla del gait), y
    llama a `_apply_upright_pose()` cada tick. `_apply_kinematic_motion`,
    `_reset_self_physics` y el reset de `_kin_action` sin cambios. Rama gait
    intacta.
  - Nuevo helper `hold_kinematic_standing()`. `hold_standing()` (gait) sin cambio
    de comportamiento.
- `tests/test_kinematic_locomotion.py`:
  - Ajustadas dos aserciones de z (`z1 == z0` → `z1 == KINEMATIC_STAND_Z`) en los
    tests de avance: el pin de postura ahora fija z también bajo el mock (antes la
    z no se pineaba en el mock). Las aserciones de movimiento/velocidad (x,y)
    quedan sin cambios.
  - Nuevos tests: pin de z + yaw puro por acción (stand/walk/turn_left/turn_right,
    parametrizado), heading preservado tras turn, y pose comandada =
    `KINEMATIC_STANDING_POSE` (clamp) ≠ cuclilla del gait.
- `README.md`: documentada la postura erguida por tick, `KINEMATIC_STANDING_POSE`
  en la tabla de tunables, y la guía de ajuste de `KINEMATIC_STAND_Z`. Plan A
  (gait) y `STANDING_POSE` sin cambios.

## Resultados exactos

### 1) pytest completo

Baseline (antes de los cambios):
```
.venv/bin/python -m pytest -q
=> 1 failed, 90 passed
   FAILED tests/test_wbt.py::test_robot_geometry_and_flags_unchanged (pre-existente, fuera de alcance)
```

Después de los cambios:
```
.venv/bin/python -m pytest -q
=> 1 failed, 96 passed
   FAILED tests/test_wbt.py::test_robot_geometry_and_flags_unchanged (MISMO fallo pre-existente)
```
+6 tests nuevos pasan; NINGÚN fallo nuevo. El único fallo sigue siendo el de
`test_wbt.py` (el `.wbt` commiteado tiene la posición de DQN desplazada; está
PROHIBIDO tocar el `.wbt`, fuera de alcance).

### 2) stable-baselines3 check_env en AMBOS modos

```
.venv/bin/python -m pytest -q "tests/test_race_env.py::test_check_env_both_modes[gait]"
=> 1 passed
.venv/bin/python -m pytest -q "tests/test_race_env.py::test_check_env_both_modes[kinematic]"
=> 1 passed
```
`check_env` (stable_baselines3.common.env_checker.check_env, warn=True,
skip_render_check=True) pasa en modo gait y en modo kinematic.

### 3) py_compile de los archivos tocados

```
.venv/bin/python -m py_compile \
  controllers/darwin_common/config.py \
  controllers/darwin_common/darwin_robot.py \
  controllers/locomotion_test/locomotion_test.py
=> OK (sin error)
```

### Verificación de límites de la pose nueva

```
KINEMATIC_STANDING_POSE: OUT_OF_RANGE [] ; OK 20 motores ; KINEMATIC_STAND_Z == 0.24
```

No se lanzó Webots. La confirmación final de la postura es visual, en Webots, por
el usuario (si el torso flota/se hunde, ajustar `KINEMATIC_STAND_Z`).
