"""Capa de robot compartida para el DARwIn-OP en la carrera.

`DarwinRobot` envuelve un Supervisor de Webots (o su mock inyectado) y expone
acciones de alto nivel (walk_forward, turn_left, turn_right, stand) más
consultas de estado (posición, orientación, caída) y utilidades de escena
(find_others, find_barriers). La locomoción se delega a un
`_LocomotionBackend` que intenta usar los managers compilados de Robotis y, si
no están disponibles, cae en un stub documentado para que los tests headless y
los entornos sin Webots funcionen.

Hecho de diseño crítico: los 4 robots comparten DEF 'DARWIN', por lo que el
robot propio se obtiene con getSelf() y los demás nodos se encuentran
escaneando los hijos de la raíz y comparando el campo 'name' (nunca getFromDef).
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

try:  # config como módulo del paquete o como módulo suelto (pythonpath).
    from . import config
except ImportError:  # pragma: no cover - ruta usada bajo pythonpath plano
    import config


class _LocomotionBackend:
    """Abstracción de locomoción con fallback documentado.

    Intenta `from managers import RobotisOp2GaitManager, RobotisOp2MotionManager`
    (binario compilado de Webots). Si el import o la inicialización fallan, usa
    un stub: control directo de motores / no-op, emitiendo un warning. Así el
    código corre headless (tests, CI) sin el runtime de Webots.
    """

    def __init__(self, supervisor, config_path: str = ""):
        self.supervisor = supervisor
        self.available = False
        self.gait = None
        self.motion = None
        try:
            from managers import RobotisOp2GaitManager, RobotisOp2MotionManager

            self.gait = RobotisOp2GaitManager(supervisor, config_path)
            self.motion = RobotisOp2MotionManager(supervisor, config_path)
            self.available = True
        except Exception as exc:  # ImportError o fallo de init
            warnings.warn(
                "Managers de Robotis no disponibles (%s); usando locomoción "
                "stub (control directo / no-op)." % exc,
                RuntimeWarning,
                stacklevel=2,
            )
            self.available = False

    def walk_forward(self) -> None:
        if self.available:
            self.gait.setXAmplitude(1.0)
            self.gait.setAAmplitude(0.0)
            self.gait.start()
        # Encola la acción para el modelo cinemático del backend (mock). En
        # Webots real este queue_action no existe y la física del gait manager
        # mueve el robot; el _queue es no-op sin romper nada.
        _queue(self.supervisor, "walk_forward")

    def turn_left(self) -> None:
        if self.available:
            self.gait.setXAmplitude(0.0)
            self.gait.setAAmplitude(0.5)
            self.gait.start()
        _queue(self.supervisor, "turn_left")

    def turn_right(self) -> None:
        if self.available:
            self.gait.setXAmplitude(0.0)
            self.gait.setAAmplitude(-0.5)
            self.gait.start()
        _queue(self.supervisor, "turn_right")

    def stand(self) -> None:
        if self.available:
            self.gait.setXAmplitude(0.0)
            self.gait.setAAmplitude(0.0)
            self.gait.stop()
        _queue(self.supervisor, None)


def _queue(supervisor, action: Optional[str]) -> None:
    """Encola una acción cinemática en el backend si éste lo soporta."""
    if hasattr(supervisor, "queue_action"):
        supervisor.queue_action(action)


class DarwinRobot:
    """Robot DARwIn-OP de alto nivel sobre un Supervisor inyectado.

    Parameters
    ----------
    backend:
        Supervisor de Webots real o su mock. Debe exponer getSelf(),
        getRoot(), step()/getBasicTimeStep() y sensores.
    name:
        Nombre del robot (debe coincidir con el campo 'name' en el .wbt).
    """

    def __init__(self, backend, name: str, config_path: str = ""):
        self.backend = backend
        self.name = name
        self.time_step = int(backend.getBasicTimeStep())
        self.locomotion = _LocomotionBackend(backend, config_path)
        self._self_node = backend.getSelf()
        self._accel = self._safe_sensor("getAccelerometer", "Accelerometer")
        self._gyro = self._safe_sensor("getGyro", "Gyro")

    # --- sensores -----------------------------------------------------------
    def _safe_sensor(self, getter_name: str, device_name: str):
        getter = getattr(self.backend, getter_name, None)
        sensor = None
        if getter is not None:
            try:
                sensor = getter(device_name)
            except Exception:
                sensor = None
        if sensor is not None and hasattr(sensor, "enable"):
            try:
                sensor.enable(self.time_step)
            except Exception:
                pass
        return sensor

    # --- acciones -----------------------------------------------------------
    def walk_forward(self) -> None:
        self.locomotion.walk_forward()

    def turn_left(self) -> None:
        self.locomotion.turn_left()

    def turn_right(self) -> None:
        self.locomotion.turn_right()

    def stand(self) -> None:
        self.locomotion.stand()

    def tick(self) -> int:
        """Avanza un paso de simulación en el backend."""
        return self.backend.step(self.time_step)

    # --- estado -------------------------------------------------------------
    def get_position(self) -> Tuple[float, float, float]:
        trans = self._self_node.getField("translation").getSFVec3f()
        return float(trans[0]), float(trans[1]), float(trans[2])

    def get_orientation(self) -> float:
        """Heading (rad) alrededor de z. 0 = mirando hacia +y (eje de carrera)."""
        rot = self._self_node.getField("rotation").getSFRotation()
        axis_z = rot[2]
        angle = rot[3]
        return float(angle if axis_z >= 0 else -angle)

    def is_fallen(self) -> bool:
        """True si la inclinación del torso supera el umbral de config.

        Usa el acelerómetro: de pie lee ~(0,0,g); la inclinación proyecta la
        gravedad en el plano horizontal. tilt = atan2(|(ax,ay)|, az).
        """
        if self._accel is None:
            return False
        ax, ay, az = self._accel.getValues()
        horizontal = math.hypot(ax, ay)
        tilt = math.atan2(horizontal, abs(az) if az != 0 else 1e-6)
        return tilt > config.FALL_TILT_THRESHOLD

    # --- pose ---------------------------------------------------------------
    def reset_pose(self, x: float, y: float, z: Optional[float] = None) -> None:
        """Reubica el robot en (x, y[, z]) mirando hacia la meta (heading 0)."""
        trans_field = self._self_node.getField("translation")
        current = trans_field.getSFVec3f()
        new_z = current[2] if z is None else z
        trans_field.setSFVec3f([float(x), float(y), float(new_z)])
        self._self_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])

    # --- escena -------------------------------------------------------------
    def _iter_world_nodes(self):
        root = self.backend.getRoot()
        children = root.getField("children")
        count = children.getCount()
        for i in range(count):
            node = children.getMFNode(i)
            if node is None:
                continue
            yield node

    @staticmethod
    def _node_name(node) -> Optional[str]:
        getter = getattr(node, "getNameString", None)
        if getter is not None:
            try:
                return getter()
            except Exception:
                pass
        name_field = None
        try:
            name_field = node.getField("name")
        except Exception:
            return None
        if name_field is None:
            return None
        try:
            value = name_field.getSFString()
        except Exception:
            value = getattr(name_field, "_value", None)
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
        return value

    @staticmethod
    def _node_xy(node) -> Optional[Tuple[float, float]]:
        try:
            trans = node.getField("translation").getSFVec3f()
        except Exception:
            return None
        return float(trans[0]), float(trans[1])

    def find_others(self) -> Dict[str, Tuple[float, float]]:
        """Otros robots por nombre (excluye self). Nunca usa getFromDef."""
        robot_names = set(config.START_POSITIONS.keys())
        others: Dict[str, Tuple[float, float]] = {}
        for node in self._iter_world_nodes():
            name = self._node_name(node)
            if name is None or name == self.name or name not in robot_names:
                continue
            xy = self._node_xy(node)
            if xy is not None:
                others[name] = xy
        return others

    def find_barriers(self) -> Dict[str, Tuple[float, float]]:
        """Vallas/obstáculos por nombre (los nodos cuyo nombre empieza por 'valla')."""
        barriers: Dict[str, Tuple[float, float]] = {}
        for node in self._iter_world_nodes():
            name = self._node_name(node)
            if name is None or not name.startswith("valla"):
                continue
            xy = self._node_xy(node)
            if xy is not None:
                barriers[name] = xy
        return barriers
