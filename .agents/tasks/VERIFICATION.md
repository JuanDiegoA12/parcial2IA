# Verificación — Gait sinusoidal Python para DARwIn-OP

Entorno: SIN runtime de Webots en el sandbox. Verificación por tests unitarios,
`check_env` de stable-baselines3 y `py_compile`. La validación física de la
caminata la hace el usuario en Webots (documentado en el README).

## Comandos ejecutados y resultados

1. **Suite completa (venv del proyecto):**
   `./.venv/bin/python -m pytest -q`
   Resultado: **58 passed, 1 failed**.
   - Antes de este trabajo: `50 passed, 1 failed`. Se añadieron 8 tests de gait.
   - El ÚNICO fallo es `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`,
     preexistente: el usuario reposicionó los robots dentro de Webots, por lo que
     `pista_atletismo_50m.wbt` tiene al DQN en `(0.21881, 9.95112, 0.22402)` en
     vez de `(0.11, 9.95, 0.24)`. Corregirlo exigiría editar la geometría del
     robot (prohibido). No lo causa esta tarea.

2. **check_env de stable-baselines3 contra el backend mock:**
   `./.venv/bin/python -m pytest -q tests/test_race_env.py::test_check_env_sb3`
   Resultado: **1 passed** (no se saltó; SB3 está instalado).

3. **py_compile de los controladores nuevos/tocados:**
   `./.venv/bin/python -m py_compile controllers/locomotion_test/locomotion_test.py`
   `controllers/dqn_agent/dqn_agent.py controllers/darwin_common/darwin_robot.py`
   `controllers/darwin_common/config.py controllers/ppo_agent/ppo_agent.py`
   `controllers/a2c_agent/a2c_agent.py controllers/custom_agent/custom_agent.py`
   Resultado: OK (todos compilan).

## Tests de gait añadidos (tests/test_darwin_robot.py)

- `test_clamp_joint_keeps_margin`: el recorte respeta `[min+margen, max-margen]`.
- `test_gait_targets_stay_within_safe_limits`: barriendo un ciclo completo, ningún
  ángulo comandado sale del rango seguro de ninguna articulación.
- `test_gait_phase_advances_on_tick`: `tick()` avanza la fase del gait.
- `test_left_and_right_legs_are_180_out_of_phase`: piernas izquierda/derecha a 180°.
- `test_stand_sets_nominal_pose`: sin amplitudes el gait equivale a la pose de pie.
- `test_actions_set_x_and_a_amplitudes`: walk_forward/turn_left/turn_right/stand
  fijan las amplitudes x/a esperadas.
- `test_is_fallen_true_on_low_height`: caída detectada por altura baja del torso.
- `test_is_fallen_true_on_strong_tilt`: caída detectada por inclinación fuerte.

## Hallazgo sobre las advertencias de rodilla (Task 1)

Confirmado por lectura del código y razonamiento del modelo: el stub anterior solo
llamaba a `supervisor.queue_action(...)`, método inexistente en Webots real, así
que nada accionaba los motores y el robot se desplomaba, dejando las rodillas
(`LegLowerR/L`, rango `[-2.25, 2.25]`) apoyadas contra su tope — de ahí las
advertencias `too big/low requested position: 2.25`. No las causaba el código
comandando ±2.25. El nuevo gait acciona los motores cada paso con recorte de
margen, por lo que las articulaciones no reposan contra sus topes. La confirmación
visual final (que las advertencias cesan con el gait activo) la realiza el usuario
en Webots con el controlador `locomotion_test` (ver README).
