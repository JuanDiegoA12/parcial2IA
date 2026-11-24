# Implementation Plan — Postura ERGUIDA del modo cinemático (Plan B)

Corregir SOLO la POSTURA VISUAL del arrastre cinemático (Plan B): hoy el robot
se desliza ACOSTADO BOCA ARRIBA (torso plano, piernas al aire) y parece levitar,
en vez de deslizarse DE PIE. El PNG del proyecto (`pista_atletismo_50m.png`) lo
confirma: el torso aparece tumbado con las rodillas flexionadas en el aire (la
cuclilla del gait aplicada en horizontal).

Todo lo demás del modo cinemático YA es correcto y se conserva byte-for-byte en
comportamiento: velocidad ~1 m/s (`KINEMATIC_LINEAR_STEP`), bloqueo de colisión
contra vallas y otros robots, `is_fallen()` siempre `False`, y el pin de z a
`0.24`. El modo gait (Plan A) queda COMPLETAMENTE intacto.

No hay runtime de Webots aquí: la verificación es `pytest` con el mock
(`tests/conftest.py`), `check_env` de stable-baselines3 y `py_compile`. La
postura final la confirma el usuario visualmente en Webots.

---

## Causa raíz (confirmada leyendo el código)

1. En `_LocomotionBackend._apply_kinematic_motion` (darwin_robot.py) la
   orientación yaw-pura + el pin de z se escriben **sólo** en la rama
   `walk_forward`. En `stand`/`turn_left`/`turn_right` no se toca la pose del
   nodo, así que entre ticks la física deja caer/tumbar el torso.
2. En `update()` modo kinematic se llama a `hold_standing()`, que comanda a los
   motores la `STANDING_POSE` del gait (una **cuclilla**), por lo que las piernas
   se ven dobladas/plegadas aunque el modo sea cinemático.
3. `resetPhysics()` se aplica pero no corrige ni la orientación ni la pose de
   piernas, así que la corrección de postura nunca llega al nodo salvo en el
   avance.

## Sutileza clave mock/tests (de la que depende el diseño)

- Bajo el mock (`FakeSupervisor`), `_apply_kinematic_motion` **retorna temprano**
  porque el supervisor expone `queue_action` (el movimiento x,y/θ lo produce
  `FakeSupervisor._apply_kinematics` en `step()`). Por tanto, cualquier escritura
  de pose que metamos DENTRO de esa guarda `queue_action` NO será observable bajo
  el mock.
- En cambio, el comando de motores `apply_targets` SÍ es observable bajo el mock
  (`supervisor._motors[name].position`), y las escrituras directas sobre el nodo
  propio (`node.getField("translation")/("rotation")`) también lo son porque
  `FakeNode` expone esos campos.

**Decisión de diseño que resuelve la sutileza:** separar la **escritura de postura
erguida** (rotación yaw-pura + pin de z sobre el nodo propio) del **arrastre de
movimiento** (x,y / θ). El movimiento mantiene su guarda `queue_action` actual
(para no duplicarlo bajo el mock). La POSTURA se escribe en un método nuevo
`_apply_upright_pose()` que se llama CADA tick en `update()` modo kinematic, para
TODAS las acciones, SIN la guarda `queue_action`, operando directamente sobre los
campos `translation`/`rotation` del nodo propio. Así:
  - el movimiento x,y/θ sigue viniendo del mock (no se duplica);
  - la corrección de postura (rotación yaw-pura, z pin) ES observable bajo el mock
    y se verifica para cada acción;
  - en Webots real ambos caminos coexisten sin pisarse (el movimiento escribe x,y;
    la postura escribe rotación yaw-pura y z, preservando la x,y vigente).

Esto exige quitar el pin de z que hoy vive en la rama `walk_forward` de
`_apply_kinematic_motion` (pasa a `_apply_upright_pose`), dejando esa rama
escribiendo SOLO x,y. El resultado numérico para el usuario es idéntico (z sigue
fijándose a `KINEMATIC_STAND_Z` cada tick), pero ahora para TODAS las acciones.

## Decisiones de diseño (breves)

- **Pose erguida como constante nueva `KINEMATIC_STANDING_POSE`** en config.py,
  separada de la `STANDING_POSE` del gait (que NO se toca). En kinematic se
  comanda ESA pose a los motores cada tick en vez de la cuclilla del gait.
- **Ángulos elegidos:** piernas casi rectas con rodilla sólo levemente flexionada
  (para que apoye de pie y no quede "bloqueada"), caderas/tobillos neutros, brazos
  caídos a los lados, cuello/cabeza neutros. Respetan los límites REALES
  asimétricos y espejados de `JOINT_LIMITS` con holgura ≥ `JOINT_LIMIT_MARGIN`
  (0.05), de modo que el recorte nunca dispara. Son una APROXIMACIÓN visual que el
  usuario puede afinar y que NO afecta el movimiento (que es cinemático).
- **`KINEMATIC_STAND_Z` se queda en 0.24** por defecto (coincide con la z de
  arranque de DQN y el rango documentado), con un comentario ampliado que guía al
  usuario a subirlo si el torso parece flotar o bajarlo si se hunde (el DARwIn-OP
  de pie ronda ~0.32–0.34 al origen del pecho según el nodo; los robots de pie del
  .wbt se sitúan ~0.24–0.34). Queda ajustable, no se adivina a ciegas.
- **`resetPhysics()` cada tick** se conserva, con su guarda defensiva real-vs-mock
  (el `FakeNode` no tiene `resetPhysics`), ya presente en `_reset_self_physics`.

## Lo que NO debe cambiar (invariantes)

- La velocidad del arrastre (`KINEMATIC_LINEAR_STEP`, `KINEMATIC_ANGULAR_STEP`) y
  la fórmula de avance `(-sin θ·d, +cos θ·d)` con la convención de heading
  (heading 0 mira a +y; `theta = rot[3] si rot[2] >= 0 else -rot[3]`;
  rotación `[0,0,1,theta]`).
- El bloqueo de colisión (`_kinematic_blocked_forward`, `KINEMATIC_COLLISION_RADIUS`,
  `find_barriers`/`find_others`), incluido que el giro NUNCA se bloquea.
- `is_fallen()` → `False` en kinematic; `get_orientation()` devuelve el mismo
  heading con los mismos signos.
- Todo el camino del gait (Plan A): `_GaitEngine`, `STANDING_POSE`,
  `compute_targets`, `standing_pose`, `hold_standing` en modo gait, la detección
  de caída en gait.
- Los agentes (`dqn_agent`, `ppo_agent`, `a2c_agent`, `custom_agent`),
  `race_env`, los experimentos, `worlds/pista_atletismo_50m.wbt` y
  `requirements.md`. NO tocar ninguno.

## Nota sobre el estado actual de los tests (PRE-EXISTENTE, fuera de alcance)

Antes de empezar, `pytest -q` da **90 passed, 1 failed**. El único fallo es
`tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`: el `.wbt` COMMITEADO
tiene la posición de DQN desplazada a `(0.21881, 9.95112, 0.22402)` en vez de
`(0.11, 9.95, 0.24)` (el robot se guardó tras correr en Webots). Este fallo NO lo
causa esta tarea y NO se corrige aquí (está PROHIBIDO tocar el `.wbt`). El criterio
de éxito de cada item de verificación es: **los tests cinemáticos pasan y no
aparece NINGÚN fallo nuevo**; el único fallo tolerado es ese `test_wbt.py`
pre-existente. Confírmalo ejecutando primero la suite para registrar el baseline.

---

## Pasos de implementación

- [ ] 0. Registrar el baseline de tests antes de cambiar nada.
      Files: (ninguno — sólo ejecución)
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q`
      — se espera `90 passed, 1 failed`, siendo el ÚNICO fallo
      `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged` (pre-existente,
      por la deriva del `.wbt`). Anota este baseline; los items siguientes no deben
      introducir fallos nuevos.

- [ ] 1. Añadir `KINEMATIC_STANDING_POSE` a config.py: dict motor→ángulo (rad) de
      una pose ERGUIDA (piernas casi rectas con rodilla levemente flexionada;
      caderas/tobillos neutros; brazos a los lados; cuello/cabeza neutros), con
      docstring en español que explique que es una APROXIMACIÓN visual ajustable,
      que respeta los límites reales espejados de `JOINT_LIMITS` con margen, y que
      NO afecta el movimiento (cinemático). Colocarla junto a `KINEMATIC_STAND_Z`,
      en la sección del Plan B. Usar EXACTAMENTE estos valores (todos dentro de
      `JOINT_LIMITS` con holgura ≥ 0.05):
      ```python
      KINEMATIC_STANDING_POSE: Dict[str, float] = {
          "ShoulderR": 0.0, "ShoulderL": 0.0,
          "ArmUpperR": 0.0, "ArmUpperL": 0.0,
          "ArmLowerR": 0.0, "ArmLowerL": 0.0,
          "PelvYR": 0.0, "PelvYL": 0.0,
          "PelvR": 0.0, "PelvL": 0.0,
          "LegUpperR": 0.0, "LegUpperL": 0.0,   # cadera neutra (pierna recta)
          "LegLowerR": 0.1, "LegLowerL": -0.1,  # rodilla: leve flexión, signo espejado R+/L-
          "AnkleR": 0.0, "AnkleL": 0.0,         # tobillo neutro
          "FootR": 0.0, "FootL": 0.0,
          "Neck": 0.0, "Head": 0.0,
      }
      ```
      (Verificación de límites: `LegUpperR∈[-1.77,0.45]`, `LegUpperL∈[-0.5,1.68]`,
      `LegLowerR∈[-0.02,2.25]`, `LegLowerL∈[-2.25,0.03]`, `ArmUpperR∈[-0.68,2.30]`,
      `ArmUpperL∈[-2.25,0.77]` — todos dejan 0.0/±0.1 con margen ≥0.05.)
      Files: controllers/darwin_common/config.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -c "import sys; sys.path.insert(0,'controllers/darwin_common'); import config; m=config.JOINT_LIMIT_MARGIN; bad=[(n,a) for n,a in config.KINEMATIC_STANDING_POSE.items() if n in config.JOINT_LIMITS and not (config.JOINT_LIMITS[n][0]+m <= a <= config.JOINT_LIMITS[n][1]-m)]; print('OUT_OF_RANGE', bad); assert not bad; print('OK', len(config.KINEMATIC_STANDING_POSE))"`
      — imprime `OK 20` y ningún ángulo fuera de rango.

- [ ] 2. Ampliar el comentario de `KINEMATIC_STAND_Z` en config.py (SIN cambiar su
      valor 0.24): añadir guía de ajuste — si el torso parece flotar, SUBIR
      `KINEMATIC_STAND_Z`; si se hunde, BAJARLO; el DARwIn-OP de pie ronda
      ~0.32–0.34 al origen del pecho según el nodo, y los robots de pie del `.wbt`
      se sitúan ~0.24–0.34. Dejar claro que es ajustable y no se adivina a ciegas.
      Files: controllers/darwin_common/config.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m py_compile controllers/darwin_common/config.py`
      — compila sin error; `KINEMATIC_STAND_Z` sigue siendo `0.24`.

- [ ] 3. En `_LocomotionBackend` (darwin_robot.py) añadir un método
      `_apply_upright_pose()` que escriba la POSTURA erguida del nodo propio cada
      tick, para TODAS las acciones, SIN la guarda `queue_action`:
      - obtener el nodo con `getSelf()` (guardas defensivas: `getSelf` ausente →
        return; nodo `None` → return);
      - leer `rotation`, derivar `theta = rot[3] if rot[2] >= 0 else -rot[3]`
        (misma convención que `get_orientation`/`_apply_kinematic_motion`) y
        reescribir `rotation = [0.0, 0.0, 1.0, float(theta)]` (yaw puro, roll/pitch
        = 0) — así `get_orientation()` devuelve el MISMO heading (los signos no
        cambian);
      - leer `translation`, conservar x,y y fijar `z = config.KINEMATIC_STAND_Z`,
        reescribiendo `translation = [x, y, KINEMATIC_STAND_Z]`;
      - envolver las lecturas/escrituras en `try/except Exception: pass` (defensivo,
        como el resto del backend).
      Este método NO mueve x,y (eso lo hace el movimiento); sólo endereza y fija z.
      Files: controllers/darwin_common/darwin_robot.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m py_compile controllers/darwin_common/darwin_robot.py`
      — compila sin error.

- [ ] 4. En `_apply_kinematic_motion` (darwin_robot.py) quitar el pin de z de la
      rama `walk_forward`: esa rama pasa a escribir SOLO x,y
      (`trans_field.setSFVec3f([x, y, _z])` conservando la z vigente, que
      `_apply_upright_pose` ya fija a `KINEMATIC_STAND_Z` el mismo tick). Mantener
      intactas la guarda `queue_action`, la fórmula de avance `(-sin θ·d, +cos θ·d)`
      y las ramas `turn_left`/`turn_right`. Actualizar el docstring para decir que
      el pin de z/yaw ahora lo hace `_apply_upright_pose`.
      Files: controllers/darwin_common/darwin_robot.py
      Verify: tests cinemáticos existentes del camino real-Webots siguen pasando:
      `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q tests/test_kinematic_locomotion.py::test_kinematic_translation_pins_stand_z_and_upright`
      — pasa (la z sigue quedando en `KINEMATIC_STAND_Z` y la rotación yaw-pura).

- [ ] 5. En `_LocomotionBackend.update()` modo kinematic, reemplazar la pose que se
      comanda a los motores y añadir la escritura de postura: (a) comandar
      `KINEMATIC_STANDING_POSE` (recortada con `self.clamp`) a los motores en vez
      de `hold_standing()` (que comanda la cuclilla del gait) — p.ej.
      `self.apply_targets({n: self.clamp(n, a) for n, a in config.KINEMATIC_STANDING_POSE.items()})`
      o un helper `hold_kinematic_standing()`; (b) llamar a `_apply_upright_pose()`
      cada tick ANTES o DESPUÉS del movimiento (el orden no importa: el movimiento
      sólo toca x,y y la postura sólo toca z/rotación); (c) conservar
      `_apply_kinematic_motion(self._kin_action)`, `_reset_self_physics()` y el
      `self._kin_action = None` final. La rama del modo gait queda intacta (sigue
      llamando `engine.advance()` + `apply_targets(compute_targets())`), así como
      `hold_standing()` para el gait (`stabilize`, `reset_pose`).
      Files: controllers/darwin_common/darwin_robot.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q tests/test_kinematic_locomotion.py tests/test_darwin_robot.py`
      — todos pasan (movimiento, giro, bloqueo e `is_fallen` del modo kinematic
      intactos, y el gait sin regresión).

- [ ] 6. Añadir tests nuevos en tests/test_kinematic_locomotion.py que observen la
      postura erguida y la pose de motores BAJO EL MOCK (que SÍ ve la escritura de
      postura porque `_apply_upright_pose` no tiene la guarda `queue_action`):
      - **Postura por acción:** para CADA acción (`stand` vía no comandar nada /
        `walk_forward` / `turn_left` / `turn_right`), partir de una z colapsada
        (p.ej. `node.set_translation([x, y, 0.075])`) y una rotación con roll/pitch
        (p.ej. `node.set_rotation([1.0, 0.0, 0.0, 0.5])`), ejecutar un
        `robot.<accion>()` + `robot.tick()`, y afirmar tras el tick:
        `z == pytest.approx(config.KINEMATIC_STAND_Z)` y la rotación es yaw-pura
        (`rot[0]==approx(0.0)`, `rot[1]==approx(0.0)`, `rot[2]==approx(1.0)`).
      - **Heading preservado:** tras `turn_left`+`tick`, `get_orientation()` sigue
        aumentando en `KINEMATIC_ANGULAR_STEP` (la reescritura yaw-pura NO altera el
        signo/heading).
      - **Pose de motores = KINEMATIC_STANDING_POSE (clamp), NO la cuclilla del
        gait:** tras un `tick` en kinematic, leer `supervisor._motors[name].position`
        y afirmar que para motores clave (p.ej. `LegLowerR`, `LegUpperR`, `AnkleR`)
        equivale a `robot.locomotion.clamp(name, config.KINEMATIC_STANDING_POSE[name])`
        y es DISTINTA del valor de la cuclilla del gait `config.STANDING_POSE[name]`
        (asegurando que ya no se comanda la cuclilla). Resolver el nombre del motor
        vía `supervisor.getMotor(name)` antes de leer `.position` si el mock crea
        motores perezosamente.
      Reutilizar el fixture `_kinematic_mode` y `_make_robot`; no se necesita
      `_make_robot_real_webots` para estos (la postura es observable bajo el mock).
      Files: tests/test_kinematic_locomotion.py
      Verify: `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q tests/test_kinematic_locomotion.py`
      — todos los tests (previos + nuevos) pasan.

- [ ] 7. Ejecutar la suite completa y `check_env`/`py_compile` para confirmar que no
      hay regresión global.
      Files: (ninguno — sólo ejecución)
      Verify:
      1) `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m pytest -q`
         — mismo resultado que el baseline salvo los tests nuevos: NINGÚN fallo
         nuevo; el único fallo sigue siendo `tests/test_wbt.py::test_robot_geometry_and_flags_unchanged`
         (pre-existente). Los tests de `check_env` de sb3 que ya existen
         (`tests/test_sb3_agents_smoke.py`, `tests/test_race_env.py`) pasan.
      2) `"/home/juan/Documentos/parcial2 IA/.venv/bin/python" -m py_compile controllers/darwin_common/config.py controllers/darwin_common/darwin_robot.py controllers/locomotion_test/locomotion_test.py`
         — compila sin error.

- [ ] 8. Actualizar README.md (prosa en español, identificadores en inglés):
      en la sección "Modo kinematic refinado" y la tabla de tunables, documentar
      que la postura ahora se escribe CADA tick para TODAS las acciones (rotación
      yaw-pura + z pin), que en kinematic se comanda `KINEMATIC_STANDING_POSE` (pose
      ERGUIDA de aproximación visual, piernas casi rectas) en vez de la cuclilla
      `STANDING_POSE` del gait, y añadir `KINEMATIC_STANDING_POSE` a la tabla de
      tunables del Plan B con la guía de ajuste de `KINEMATIC_STAND_Z` (subir si
      flota, bajar si se hunde). Dejar claro que el gait (Plan A) y `STANDING_POSE`
      no cambian. NO tocar `requirements.md`.
      Files: README.md
      Verify: revisión manual del diff (`git diff README.md`) — menciona
      `KINEMATIC_STANDING_POSE`, la escritura de postura por acción y la guía de
      `KINEMATIC_STAND_Z`; no hay cambios en la sección del Plan A gait.

- [ ] 9. Commit local en la rama `master` (NUNCA push/fetch/pull). Añadir SOLO los
      archivos tocados por esta tarea; NO incluir `worlds/pista_atletismo_50m.wbt`
      (modificado fuera de alcance), `requirements.md`, ni el `.png`.
      Files: controllers/darwin_common/config.py,
      controllers/darwin_common/darwin_robot.py,
      tests/test_kinematic_locomotion.py, README.md
      Verify:
      `git add controllers/darwin_common/config.py controllers/darwin_common/darwin_robot.py tests/test_kinematic_locomotion.py README.md`
      y luego `git commit -m "fix(kinematic): postura erguida de pie en Plan B (pose y orientación por tick)"`;
      confirmar con `git show --stat HEAD` que el commit incluye SÓLO esos 4
      archivos y que `git status --short` ya no lista esos 4 como modificados
      (pueden seguir listados `worlds/pista_atletismo_50m.wbt` y el `.png` como no
      rastreados/modificados: quedan fuera de alcance a propósito).
