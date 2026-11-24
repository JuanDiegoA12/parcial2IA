"""Controlador de PRUEBA de locomoción para el DARwIn-OP (sin RL).

Valida el gait sinusoidal de `darwin_common` sobre un único robot, SIN
stable-baselines3 ni ningún algoritmo de aprendizaje. Útil para comprobar en
Webots que el robot se mantiene de pie, avanza y gira antes de integrar los
cuatro agentes.

Uso en Webots:
    1. Asigna este controlador a un robot DARwIn-OP del mundo (campo
       `controller "locomotion_test"`).
    2. Elige el comando con la variable de entorno LOCO_CMD:
           LOCO_CMD=stand    -> mantiene la pose de pie (por defecto)
           LOCO_CMD=forward  -> camina hacia adelante
           LOCO_CMD=left     -> gira a la izquierda
           LOCO_CMD=right    -> gira a la derecha
       Ejemplo: `LOCO_CMD=forward webots ./pista_atletismo_50m.wbt`
    3. Observa en la consola el log de posición / heading / altura /
       acelerómetro en cada paso.

Hallazgo sobre las advertencias de rodilla (confirmación empírica — Task 1):
    Con el stub anterior nada accionaba los motores en Webots (su único efecto
    era `supervisor.queue_action(...)`, método inexistente en Webots real), de
    modo que el robot se desplomaba y sus articulaciones quedaban apoyadas
    contra los topes. Las advertencias
        RotationalMotor "LegLowerR/L": too big/low requested position: 2.25
    provienen de ese COLAPSO contra el límite de la rodilla ([-2.25, 2.25]),
    NO de que el código comandara ±2.25. Al mantener este controlador la pose
    de pie y accionar el gait cada paso (con recorte a [min+margen, max-margen]),
    las articulaciones dejan de reposar contra sus topes y las advertencias
    desaparecen mientras el gait sostiene las piernas.
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


def _read_command() -> str:
    """Lee LOCO_CMD (stand|forward|left|right); por defecto 'stand'."""
    value = os.environ.get("LOCO_CMD", "stand").strip().lower()
    if value not in _COMMANDS:
        print("[locomotion_test] LOCO_CMD='%s' no reconocido; uso 'stand'." % value)
        value = "stand"
    return value


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
    print("[locomotion_test] robot=%s comando=%s" % (robot_name, command))
    print("[locomotion_test] motores resueltos: %d/%d" % (
        len(robot.locomotion.motors), len(config.MOTOR_NAMES)))

    # Pose de pie + estabilización inicial (asienta la física).
    robot.reset_pose(*robot.get_position())

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
