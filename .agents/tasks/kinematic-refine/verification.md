# Verificación — Refinar el modo cinemático (Plan B)

Entorno SIN Webots (tests con mock en memoria de la API de Webots). Intérprete:
`.venv/bin/python`. Rama: `master`. No se hizo push/fetch/pull.

## Comandos ejecutados y resultados

### 1. Suite completa de pytest

```
.venv/bin/python -m pytest -q
```

Resultado: **90 passed, 1 failed in ~24 s**.

- El ÚNICO fallo es el pre-existente y FUERA DE ALCANCE:
  `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`
  (el usuario reposicionó robots en el .wbt a mano; no se toca la geometría).
- Baseline antes del cambio: 85 passed, 1 failed. Tras el refinamiento:
  90 passed, 1 failed (se añadieron 5 tests nuevos del modo kinematic refinado:
  avance por `config.KINEMATIC_LINEAR_STEP`, valla delante bloquea, giro sigue
  funcionando bloqueado, otro robot delante bloquea, y pin de z + yaw puro en el
  camino real-Webots).

### 2. check_env de stable-baselines3 en AMBOS modos

```
LOCOMOTION_MODE=kinematic .venv/bin/python -c "... check_env(DarwinRaceEnv(...)) ..."
LOCOMOTION_MODE=gait      .venv/bin/python -c "... check_env(DarwinRaceEnv(...)) ..."
```

Resultado:
```
mode=kinematic robot.mode=kinematic
check_env OK for mode=kinematic
mode=gait robot.mode=gait
check_env OK for mode=gait
```

Ambos modos pasan `check_env`. Además, dentro de la suite,
`tests/test_race_env.py::test_check_env_both_modes[gait]` y `[kinematic]` pasan.

### 3. py_compile de los archivos cambiados

```
.venv/bin/python -m py_compile \
  controllers/darwin_common/config.py \
  controllers/darwin_common/darwin_robot.py \
  tests/conftest.py \
  tests/test_kinematic_locomotion.py \
  tests/test_race_env.py \
  tests/test_race.py
```

Resultado: **py_compile OK** (sin errores).

### 4. Webots

NO se lanzó Webots (no hay runtime en este entorno), según lo indicado.

## Cambios realizados (resumen)

- `config.py`: `KINEMATIC_LINEAR_STEP` 0.25 → **0.016** (1.0 m/s a 16 ms);
  `KINEMATIC_ANGULAR_STEP` 0.20 → **0.04** (~143 °/s); nuevas constantes
  `KINEMATIC_COLLISION_RADIUS = 0.3` y `KINEMATIC_STAND_Z = 0.24`, todas
  documentadas con su derivación. Añadida nota de qué verificar en Webots.
- `darwin_robot.py`: helpers de escaneo de escena compartidos
  (`_iter_world_nodes` / `_node_name` / `_node_xy` a nivel módulo);
  `_LocomotionBackend._kinematic_blocked_forward()` + `_obstacles_xy()` +
  `_self_xy_theta()` deciden el bloqueo (vallas y otros robots) en DarwinRobot,
  no en el mock; `walk_forward()` cancela el avance cinemático si está bloqueado
  (sin encolar, para no duplicar movimiento); `_apply_kinematic_motion()` fija
  `z = KINEMATIC_STAND_Z` (pin visual de pie) manteniendo yaw puro. El camino
  del gait (Plan A) queda intacto.
- `tests/conftest.py`: el mock lee `config.KINEMATIC_LINEAR_STEP` /
  `KINEMATIC_ANGULAR_STEP` (fuente única de la velocidad); `WALK_DELTA` /
  `TURN_DELTA` ahora derivan de esas constantes.
- `tests/test_kinematic_locomotion.py`: tests (a)-(e) del modo refinado.
- `tests/test_race_env.py` y `tests/test_race.py`: metas/pasos derivados del
  nuevo `KINEMATIC_LINEAR_STEP` (ya no literales 0.25) y, en la carrera, el
  robot de prueba se reubica a un carril libre porque ahora los obstáculos
  bloquean el avance.
- `README.md`: velocidad realista (m/s), bloqueo de obstáculos (y penalización
  que sigue activa), pin de pie (`KINEMATIC_STAND_Z`) como aproximación, gait
  como opt-in sin cambios, y que la velocidad se afina en un solo lugar
  (`config.py`).
