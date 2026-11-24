"""Controlador de PRUEBA de locomoción para el DARwIn-OP (sin RL).

Valida el gait sinusoidal de `darwin_common` sobre un único robot, SIN
stable-baselines3 ni ningún algoritmo de aprendizaje. Útil para comprobar en
Webots que el robot se mantiene de pie, avanza y gira antes de integrar los
cuatro agentes.

Uso en Webots:
    1. Asigna este controlador a un robot DARwIn-OP del mundo (campo
       `controller "locomotion_test"`).
    2. Elige el comando con PRIORIDAD:
           a) el primer argumento de controllerArgs del .wbt (argv[1]), p.ej.
              `controllerArgs ["forward"]`;
           b) si no hay, la variable de entorno LOCO_CMD;
           c) si tampoco, por defecto 'stand'.
       Comandos válidos: stand | forward | left | right.
           stand    -> mantiene la pose de pie (por defecto)
           forward  -> camina hacia adelante
           left     -> gira a la izquierda
           right    -> gira a la derecha
       Ejemplos: `controllerArgs ["forward"]` en el .wbt, o
       `LOCO_CMD=forward webots ./pista_atletismo_50m.wbt`.
    3. Al arrancar, en MODO DIAGNÓSTICO el controlador imprime los límites
       REALES (min/max) de los 20 motores leídos con getMinPosition()/
       getMaxPosition(); cópialos para fijar JOINT_LIMITS si hiciera falta.
       Tras la estabilización imprime la z de pie resuelta (~0.23-0.25).
    4. Observa en la consola el log de posición / heading / altura /
       acelerómetro en cada paso.

Hallazgo sobre las advertencias de rodilla/cadera (origen REAL, no el mock):
    Las advertencias provienen del MODELO Darwin-op real de Webots (su PROTO
    remoto con rangos asimétricos y espejados), NO del stub/mock de los tests
    (el mock nunca emite esas advertencias). Hay dos causas distintas:
      1) "LegLowerR/L too big/low 2.25": con el stub anterior nada accionaba
         los motores en Webots (su único efecto era `queue_action(...)`,
         inexistente en Webots real), el robot se desplomaba y la rodilla
         quedaba apoyada contra su tope de un solo lado (2.25 en R, -2.25 en L).
      2) "LegUpperR 0.5 > 0.45": ese era un BUG NUESTRO: STANDING_POSE pedía
         0.5 cuando el tope real de LegUpperR es 0.45, y el JOINT_LIMITS
         simétrico (-1.57,1.57) de antes no lo recortaba.
    Correcciones: la pose de pie ahora queda dentro de los límites reales con
    margen; los límites reales se leen en RUNTIME (getMinPosition/getMaxPosition)
    y se usan para el recorte final; y este controlador sostiene la pose/gait
    cada paso, por lo que las articulaciones dejan de reposar contra sus topes.
"""

from __future__ import annotations

import os
import sys

# Permite importar el paquete común (config, darwin_robot) tanto bajo pytest
# como al ejecutarse como controlador de Webots.
_COMMON = os.path.join(os.path.dirname(__file__), "..", "darwin_common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

import config  # noqa: E402
from darwin_robot import DarwinRobot  # noqa: E402

# Comando -> método de locomoción de alto nivel.
_COMMANDS = {
    "stand": "stand",
    "forward": "walk_forward",
    "left": "turn_left",
    "right": "turn_right",
}


def _read_command(argv=None) -> str:
    """Resuelve el comando con prioridad controllerArgs -> LOCO_CMD -> 'stand'.

    En Webots los controllerArgs del .wbt llegan como argv; se usa argv[1] si
    existe. Si no, la variable de entorno LOCO_CMD. Si tampoco, 'stand'.
    Cualquier valor fuera de {stand,forward,left,right} cae a 'stand'.
    """
    if argv is None:
        argv = sys.argv
    raw = None
    if len(argv) > 1 and argv[1].strip():
        raw = argv[1]
    elif os.environ.get("LOCO_CMD", "").strip():
        raw = os.environ.get("LOCO_CMD")
    value = (raw or "stand").strip().lower()
    if value not in _COMMANDS:
        print("[locomotion_test] comando '%s' no reconocido; uso 'stand'." % value)
        value = "stand"
    return value


def _motor_limits(motor):
    """Lee (min, max) reales del motor de forma defensiva; None si no se puede."""
    get_min = getattr(motor, "getMinPosition", None)
    get_max = getattr(motor, "getMaxPosition", None)
    if not callable(get_min) or not callable(get_max):
        return None
    try:
        return float(get_min()), float(get_max())
    except Exception:
        return None


def _print_motor_diagnostics(robot: DarwinRobot) -> None:
    """Vuelca por consola los límites REALES de los 20 motores (una línea c/u).

    Para motores sin getMinPosition/getMaxPosition (p.ej. el mock) cae a
    config.JOINT_LIMITS o imprime 'n/a'.
    """
    print("[locomotion_test] --- límites reales de motores (min, max) ---")
    motors = robot.locomotion.motors
    for name in config.MOTOR_NAMES:
        motor = motors.get(name)
        limits = _motor_limits(motor) if motor is not None else None
        if limits is None:
            limits = config.JOINT_LIMITS.get(name)
        if limits is None:
            print("[locomotion_test]   %-10s min=n/a    max=n/a" % name)
        else:
            print("[locomotion_test]   %-10s min=%.4f max=%.4f" % (name, limits[0], limits[1]))
    print("[locomotion_test] --- fin límites de motores ---")


def _accel_values(robot: DarwinRobot):
    """Lectura del acelerómetro de forma defensiva (None si no hay sensor)."""
    accel = getattr(robot, "_accel", None)
    if accel is None:
        return None
    try:
        return accel.getValues()
    except Exception:
        return None


def main() -> None:
    from controller import Supervisor

    supervisor = Supervisor()
    robot_name = os.environ.get("LOCO_ROBOT", "DQN")
    robot = DarwinRobot(supervisor, robot_name)

    command = _read_command()
    method = _COMMANDS[command]
    mode = config.resolve_locomotion_mode()
    print("[locomotion_test] robot=%s comando=%s modo=%s" % (robot_name, command, mode))
    if mode == "kinematic":
        print("[locomotion_test] modo kinematic (Plan B): el Supervisor arrastra "
              "al robot de pie; is_fallen() siempre False, sin ajuste de marcha.")
    print("[locomotion_test] motores resueltos: %d/%d" % (
        len(robot.locomotion.motors), len(config.MOTOR_NAMES)))

    # Modo diagnóstico: volcar los límites reales de los 20 motores.
    _print_motor_diagnostics(robot)

    # Pose de pie + estabilización inicial (asienta la física).
    robot.reset_pose(*robot.get_position())

    # z de pie resuelta tras estabilizar (referencia para FALL_HEIGHT_Z y pose).
    _, _, standing_z = robot.get_position()
    print("[locomotion_test] z de pie resuelta tras estabilización: %.4f" % standing_z)

    step = 0
    while True:
        # Fijar amplitudes del comando elegido cada paso y accionar el gait.
        getattr(robot, method)()
        if robot.tick() == -1:
            break

        if step % 10 == 0:
            x, y, z = robot.get_position()
            heading = robot.get_orientation()
            accel = _accel_values(robot)
            fallen = robot.is_fallen()
            print(
                "[locomotion_test] step=%d pos=(%.3f, %.3f, %.3f) heading=%.3f "
                "accel=%s fallen=%s" % (step, x, y, z, heading, accel, fallen)
            )
        step += 1


if __name__ == "__main__":
    main()
