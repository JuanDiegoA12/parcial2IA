# Implementation Plan — Refinar el modo cinemático (Plan B)

Refinar SOLO el arrastre cinemático (Plan B) de la locomoción DARwIn-OP: velocidad
realista, colisión que BLOQUEA el avance contra vallas y otros robots, y pinchado
visual de pie (z y orientación). El gait (Plan A) queda byte-for-byte intacto.
LOCOMOTION_MODE por defecto sigue siendo "kinematic".

No hay Webots en este entorno: toda la verificación es vía pytest con el mock
(`tests/conftest.py`), `check_env` de stable-baselines3 y `py_compile`. Lo que el
usuario debe validar en Webots real se documenta en el código y en el item final.

## Decisiones de diseño (una o dos frases cada una)

- **Fuente única de la velocidad.** El desplazamiento cinemático real ya lee
  `config.KINEMATIC_LINEAR_STEP` / `KINEMATIC_ANGULAR_STEP` en
  `_LocomotionBackend._apply_kinematic_motion` (darwin_robot.py). Mantenemos ese
  patrón y además hacemos que el mock (`_apply_kinematics` en conftest.py) lea esas
  MISMAS constantes de config, de modo que la magnitud del movimiento sea idéntica
  bajo el mock y en Webots real y la velocidad se afine en UN solo lugar
  (config.py). Esto evita dejar tests asertando 0.25.
- **Valores elegidos (derivados a basicTimeStep = 16 ms = 0.016 s/tick):**
  - `KINEMATIC_LINEAR_STEP = 0.016` m/tick → 0.016 / 0.016 = **1.0 m/s** (paso
    humano). Antes 0.25 m/tick = ~15.6 m/s (irreal, hacía trivial la carrera).
  - `KINEMATIC_ANGULAR_STEP = 0.04` rad/tick → 0.04 / 0.016 = 2.5 rad/s ≈
    **143 °/s** (giro suave pero usable). Antes 0.20 rad/tick = 12.5 rad/s ≈ 716 °/s.
  - `KINEMATIC_COLLISION_RADIUS = 0.3` m (semiancho del robot ~0.15 + semiespesor
    de la valla ~0.15). Queda por DEBAJO de `SAFETY_RADIUS = 0.6` (la penalización
    por proximidad sigue disparándose ANTES del bloqueo) y por ENCIMA de
    `HARD_COLLISION_DISTANCE = 0.15`.
  - `KINEMATIC_STAND_Z = 0.24` m (z nominal de pie; coincide con la z de arranque de
    DQN = 0.24 y el rango documentado ~0.23–0.25).
- **Dónde se decide el bloqueo.** La decisión de bloqueo se toma en
  `_LocomotionBackend` (DarwinRobot), NO en el mock: antes de registrar un
  `walk_forward`, se calcula la (x,y) de destino con el heading actual y se consulta
  `find_barriers()` / `find_others()`. Si el destino cae dentro de
  `KINEMATIC_COLLISION_RADIUS` de algún obstáculo u otro robot, se CANCELA el
  desplazamiento (no se encola ni se registra la acción cinemática) para ese tick.
  Así el bloqueo es determinista y testeable bajo el mock, y se evita el doble
  movimiento (el mock sólo mueve si se le encoló la acción). El giro SIEMPRE se
  permite (el agente puede esquivar).
- **Pin visual de pie.** En el camino cinemático, cada escritura de `translation`
  fija z = `KINEMATIC_STAND_Z` y la `rotation` se mantiene como yaw puro sobre z
  (roll/pitch = 0), además de `hold_standing()` + `resetPhysics()` que ya existen.
  Es una aproximación visual/cinemática; se documenta qué debe verificar el usuario
  en Webots.
- **find_barriers/find_others necesitan el nodo propio.** Ambos ya funcionan con el
  nodo propio resuelto; el acceso a la posición propia para el cálculo de destino usa
  el mismo `getSelf()`/`getField("translation")` que el resto del camino cinemático.
  Bajo el mock, `find_barriers()`/`find_others()` leen el mundo compartido, así que el
  bloqueo es determinista.

## Convención de movimiento (idéntica en mock y en Webots real, NO cambiar)

heading 0 mira a +y; `theta = rot[3] if rot[2] >= 0 else -rot[3]`;
avance = `(x - sin(theta)*d, y + cos(theta)*d, z)`; `turn_left` suma, `turn_right`
resta `KINEMATIC_ANGULAR_STEP` a theta. El bloqueo debe usar ESTA misma fórmula para
calcular el destino, de modo que las aserciones de los tests sean deterministas.

---

- [ ] 1. Actualizar las constantes cinemáticas en config.py.
      En `controllers/darwin_common/config.py`, cambiar `KINEMATIC_LINEAR_STEP` de
      0.25 a `0.016` y `KINEMATIC_ANGULAR_STEP` de 0.20 a `0.04`, y REESCRIBIR sus
      comentarios para mostrar la derivación a 16 ms (1.0 m/s y ~143 °/s) y marcarlas
      AJUSTABLE. Añadir, justo después, dos constantes nuevas con comentario:
      `KINEMATIC_COLLISION_RADIUS: float = 0.3` (documentar: semiancho robot + semiespesor
      valla; < SAFETY_RADIUS para que la penalización por proximidad siga disparándose
      antes del bloqueo; > HARD_COLLISION_DISTANCE) y `KINEMATIC_STAND_Z: float = 0.24`
      (documentar: z nominal de pie; pin visual del torso en modo kinematic). NO
      eliminar ni renumerar constantes existentes; NO tocar el bloque del gait ni
      `RaceParams`.
      Files: controllers/darwin_common/config.py
      Verify: `.venv/bin/python -m pytest -q tests/test_config.py` pasa; y
      `.venv/bin/python -c "import sys; sys.path.insert(0,'controllers/darwin_common'); import config; print(config.KINEMATIC_LINEAR_STEP, config.KINEMATIC_ANGULAR_STEP, config.KINEMATIC_COLLISION_RADIUS, config.KINEMATIC_STAND_Z)"`
      imprime `0.016 0.04 0.3 0.24`.

- [ ] 2. Acoplar el mock a las constantes de config (sustituir WALK_DELTA/TURN_DELTA).
      En `tests/conftest.py`, hacer que el modelo cinemático de juguete lea las
      constantes de config en vez de los literales 0.25/0.20. Opción elegida: en
      `FakeSupervisor._apply_kinematics` usar `config.KINEMATIC_LINEAR_STEP` y
      `config.KINEMATIC_ANGULAR_STEP` directamente, y redefinir los módulos
      `WALK_DELTA = config.KINEMATIC_LINEAR_STEP` / `TURN_DELTA = config.KINEMATIC_ANGULAR_STEP`
      (conservando los nombres por si algún test los referencia) con un comentario que
      explique el acople (la velocidad se afina SOLO en config.py). Mantener la MISMA
      fórmula de avance/giro (convención de heading intacta). NO cambiar `_BARRIERS`,
      `_ROBOT_STARTS`, `force_tilt` ni `FakeMotor`.
      Files: tests/conftest.py
      Verify: `.venv/bin/python -m pytest -q tests/test_kinematic_locomotion.py` pasa
      (las aserciones leen `config.KINEMATIC_LINEAR_STEP`/`ANGULAR_STEP`, ya no 0.25/0.20),
      confirmando que el mock se mueve exactamente el nuevo paso.

- [ ] 3. Implementar bloqueo de colisión en el camino cinemático de _LocomotionBackend.
      En `controllers/darwin_common/darwin_robot.py`, dentro de `_LocomotionBackend`,
      añadir un helper privado `_kinematic_blocked_forward(self) -> bool` que: lee la
      posición y heading propios del nodo (`getSelf()`/`getField("translation")` y
      `getField("rotation")`, con la misma convención theta que
      `_apply_kinematic_motion`); calcula el destino
      `(nx, ny) = (x - sin(theta)*KINEMATIC_LINEAR_STEP, y + cos(theta)*KINEMATIC_LINEAR_STEP)`;
      y devuelve True si `hypot(nx - bx, ny - by) < KINEMATIC_COLLISION_RADIUS` para
      algún `(bx,by)` de `self._barriers_xy()`/`self._others_xy()`. Para obtener vallas y
      otros robots sin acoplar a `DarwinRobot`, resolverlos vía el supervisor: el backend ya
      tiene `self.supervisor`; reutilizar el mismo escaneo de escena que usa
      `DarwinRobot.find_barriers`/`find_others` (extraer un helper compartido de escaneo,
      o exponer en el backend una referencia al escaneo del robot). Mantenerlo simple y
      defensivo (si no hay nodo propio o falla el acceso, devolver False = no bloquea).
      Luego, en `_LocomotionBackend.walk_forward`, SOLO cuando `self.mode == "kinematic"`:
      si `_kinematic_blocked_forward()` es True, NO llamar a `_queue(..., "walk_forward")`
      y NO fijar `self._kin_action = "walk_forward"` (dejarlo en None para ese tick); el
      resto (set_amplitudes del gait) queda INTACTO. El giro (`turn_left`/`turn_right`) no
      cambia. Documentar en docstring que esto impide penetrar la valla y que el giro
      permite esquivar.
      IMPORTANTE: no duplicar el movimiento — el bloqueo evita encolar la acción, de modo
      que ni el mock (`_apply_kinematics`) ni `_apply_kinematic_motion` la aplican.
      Files: controllers/darwin_common/darwin_robot.py
      Verify: `.venv/bin/python -m pytest -q tests/test_darwin_robot.py tests/test_kinematic_locomotion.py`
      pasa (incluye los tests de gait sin cambios y los nuevos de bloqueo del item 6).

- [ ] 4. Pinchar z y orientación de pie en la escritura cinemática de translation/rotation.
      En `controllers/darwin_common/darwin_robot.py`, en
      `_LocomotionBackend._apply_kinematic_motion`: al escribir `translation` en
      `walk_forward`, fijar la componente z a `config.KINEMATIC_STAND_Z` (en vez de
      heredar la z colapsada): `trans_field.setSFVec3f([float(x), float(y), float(config.KINEMATIC_STAND_Z)])`.
      Mantener la rotación como yaw puro sobre z (ya se escribe `[0,0,1,theta]` en los
      giros; en `walk_forward` no se toca rotation, lo cual es correcto porque ya es yaw
      puro tras `reset_pose`). Añadir docstring/comentario de que esto es una
      aproximación VISUAL (roll/pitch = 0, z fija) y que el usuario debe verificar en
      Webots que el torso no se acuesta. NOTA: este camino sólo actúa en Webots real
      (guardado por `hasattr(self.supervisor, "queue_action")`), así que el mock no lo
      ejerce por esta vía; el test (e) lo ejercita invocando el método directamente sobre
      un supervisor SIN `queue_action` (ver item 6e).
      Files: controllers/darwin_common/darwin_robot.py
      Verify: `.venv/bin/python -m pytest -q tests/test_kinematic_locomotion.py` pasa
      (incluye el nuevo test (e) del item 6).

- [ ] 5. Documentar el comportamiento cinemático refinado para validación en Webots.
      En el docstring del modo kinematic en `config.py` (bloque de LOCOMOTION_MODE) y/o
      en el docstring de `_apply_kinematic_motion`/`_kinematic_blocked_forward` en
      `darwin_robot.py`, añadir 3–5 líneas (español) que enumeren qué debe verificar el
      usuario en Webots real: (1) velocidad ~1 m/s con LOCO_CMD=forward; (2) el robot se
      detiene al llegar a una valla y puede girar para esquivarla; (3) el torso se mantiene
      a z~0.24 sin acostarse; (4) is_fallen() sigue False en kinematic. Cambio puramente
      documental; identificadores en inglés, comentarios en español.
      Files: controllers/darwin_common/config.py, controllers/darwin_common/darwin_robot.py
      Verify: `.venv/bin/python -m py_compile controllers/darwin_common/config.py controllers/darwin_common/darwin_robot.py`
      sin errores.

- [ ] 6. Añadir/ajustar los tests unitarios (a–f) del modo cinemático refinado.
      En `tests/test_kinematic_locomotion.py` añadir/ajustar, reutilizando el fixture
      `_kinematic_mode` y la convención del mock:
      (a) `test_walk_forward_advances_by_config_linear_step`: a heading 0, `walk_forward`+`tick`
          avanza `config.KINEMATIC_LINEAR_STEP` en +y (leer la constante, NO 0.25). Ya existe
          equivalente (`test_walk_forward_advances_along_plus_y_at_heading_zero`); confirmar
          que asierta contra la constante.
      (b) `test_barrier_ahead_blocks_forward`: colocar el robot propio justo detrás de una
          valla a < `KINEMATIC_COLLISION_RADIUS` del destino (p.ej. reubicar el nodo propio
          a `(bx, by - 0.2)` de 'valla 1(9)' con heading 0 mirando +y hacia la valla), llamar
          `walk_forward`+`tick` y afirmar que la posición NO cambió (no penetra la valla):
          `x1,y1 == x0,y0` y que la distancia a la valla no bajó de `KINEMATIC_COLLISION_RADIUS`.
      (c) `test_turn_still_works_when_blocked`: en la misma configuración bloqueada de (b),
          `turn_left`+`tick` SÍ cambia el heading en `config.KINEMATIC_ANGULAR_STEP` (el giro
          no se bloquea).
      (d) `test_other_robot_ahead_blocks_forward`: reubicar al robot propio justo detrás de
          otro robot (usar dos nombres del mundo, p.ej. self='DQN' detrás de 'ppo') a <
          `KINEMATIC_COLLISION_RADIUS` del destino; `walk_forward`+`tick` no cambia la posición.
      (e) `test_kinematic_translation_pins_stand_z_and_upright`: construir un supervisor SIN
          `queue_action` (p.ej. `delattr`/monkeypatch para simular Webots real: el nodo propio
          con `getSelf`, `getField`, `resetPhysics` no-op, pero sin `queue_action`), fijar el
          nodo propio a una z colapsada (p.ej. 0.075), invocar `walk_forward` + un tick (o
          `locomotion._apply_kinematic_motion("walk_forward")` directamente), y afirmar que la
          z resultante == `config.KINEMATIC_STAND_Z` y que `rotation` es yaw puro
          (`rot[0]==rot[1]==0`, `rot[2]==1`). Documentar que esto cubre el camino real-Webots.
      (f) Los tests de gait existentes (`tests/test_darwin_robot.py`:
          `test_gait_phase_advances_on_tick`, `test_gait_targets_stay_within_safe_limits`,
          `test_left_and_right_legs_are_180_out_of_phase`, `test_stand_sets_nominal_pose`,
          `test_actions_set_x_and_a_amplitudes`, etc.) deben seguir pasando SIN cambios
          (camino del gait intacto). No editarlos.
      Files: tests/test_kinematic_locomotion.py (y, si hace falta un helper de supervisor
      "real-Webots" reutilizable, tests/conftest.py — explicar el añadido en comentario)
      Verify: `.venv/bin/python -m pytest -q tests/test_kinematic_locomotion.py tests/test_darwin_robot.py`
      pasa; los nuevos (b)(c)(d)(e) pasan y ningún test de gait cambia.

- [ ] 7. Verificación global del suite completo y check_env en ambos modos.
      Ejecutar todo el suite y confirmar el único fallo pre-existente esperado.
      Files: (ninguno — verificación)
      Verify:
      1) `.venv/bin/python -m pytest -q` → todo verde EXCEPTO
         `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged` (fallo conocido,
         FUERA DE ALCANCE; NO tocar la geometría del .wbt). Reportar el conteo exacto
         pasados/fallidos (baseline actual: 85 passed, 1 failed).
      2) `check_env` en ambos modos ya está cubierto por
         `tests/test_race_env.py::test_check_env_both_modes[gait]` y `[kinematic]`;
         confirmar que ambos pasan dentro del suite.
      3) `.venv/bin/python -m py_compile controllers/darwin_common/config.py controllers/darwin_common/darwin_robot.py tests/conftest.py tests/test_kinematic_locomotion.py`
         sin errores.
      4) NO lanzar Webots.

## Notas de alcance (honrar)

- NO modificar: lógica de algoritmos de agentes, `experiments/`,
  `worlds/pista_atletismo_50m.wbt` ni `pista_atletismo_50m.wbt`, `requirements.md`.
- NO corregir `test_wbt.py::test_robot_geometry_and_flags_unchanged` (el usuario cambió
  posiciones de robots en el .wbt a mano; fuera de alcance).
- Gait (Plan A) byte-for-byte intacto; firmas públicas intactas; `is_fallen()` sigue
  devolviendo False en modo kinematic.
- Commits locales solamente; NUNCA push/fetch/pull aunque exista 'origin'.
- Comentarios/README en español; identificadores en inglés. Cambios quirúrgicos y aditivos.

## Lo que el implementador debe revisar al convergir

- Confirmar que el bloqueo NO rompe `test_race_env.py` (las penalizaciones por proximidad
  y el cruce de meta deben seguir idénticos en espíritu; el bloqueo sólo cancela el
  desplazamiento, no cambia la recompensa).
- Confirmar que no se aplica el movimiento dos veces bajo el mock (bloqueo = no encolar).
