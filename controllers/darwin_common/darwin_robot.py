"""Capa de robot compartida para el DARwIn-OP en la carrera.

`DarwinRobot` envuelve un Supervisor de Webots (o su mock inyectado) y expone
acciones de alto nivel (walk_forward, turn_left, turn_right, stand) más
consultas de estado (posición, orientación, caída) y utilidades de escena
(find_others, find_barriers). La locomoción se delega a un
`_LocomotionBackend` que acciona directamente los 20 motores del DARwIn-OP con
un gait sinusoidal propio (`_GaitEngine`), porque los managers de Robotis solo
existen en C++ (`import managers` falla en Python). Todas las constantes
ajustables del gait viven en config.py.

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


def _clamp_joint(name: str, angle: float) -> float:
    """Recorta `angle` a [min+margen, max-margen] del motor `name`.

    Si el motor no tiene límites declarados se devuelve el ángulo intacto.
    Nunca se comanda una articulación exactamente a su tope físico: así se
    evitan las advertencias "too big/low requested position" de Webots.
    """
    limits = config.JOINT_LIMITS.get(name)
    if limits is None:
        return float(angle)
    lo, hi = limits
    margin = config.JOINT_LIMIT_MARGIN
    lo_safe = lo + margin
    hi_safe = hi - margin
    if lo_safe > hi_safe:  # margen mayor que el rango: usa el centro
        return float((lo + hi) / 2.0)
    return float(max(lo_safe, min(hi_safe, angle)))


class _GaitEngine:
    """Generador de patrones (CPG sencillo) para el gait sinusoidal.

    Centraliza TODA la cinemática de locomoción. Mantiene la fase del ciclo y
    dos amplitudes de comando (x = avance, a = giro) análogas a
    setXAmplitude/setAAmplitude del controlador C++ de Robotis, pero resueltas
    por nosotros. `compute_targets(phase)` devuelve un ángulo objetivo por
    motor, ya recortado a los límites seguros.
    """

    def __init__(self):
        self.phase = 0.0
        self.x_amplitude = 0.0
        self.a_amplitude = 0.0

    def set_amplitudes(self, x: float, a: float) -> None:
        self.x_amplitude = float(x)
        self.a_amplitude = float(a)

    def reset_phase(self) -> None:
        self.phase = 0.0

    def advance(self) -> None:
        """Avanza la fase un paso de simulación (un tick)."""
        self.phase += 2.0 * math.pi * config.GAIT_FREQUENCY
        # Mantener la fase acotada para evitar crecer sin límite.
        two_pi = 2.0 * math.pi
        if self.phase >= two_pi:
            self.phase -= two_pi

    def standing_pose(self) -> Dict[str, float]:
        """Pose de pie nominal, recortada a los límites seguros."""
        return {name: _clamp_joint(name, angle) for name, angle in config.STANDING_POSE.items()}

    def compute_targets(self, phase: Optional[float] = None) -> Dict[str, float]:
        """Ángulos objetivo por motor para la fase dada (o la actual).

        Parte de STANDING_POSE y suma términos sinusoidales coordinados en
        cadera (LegUpper), rodilla (LegLower) y tobillo (Ankle), con las
        piernas izquierda y derecha desfasadas 180 grados, más un balanceo
        lateral del peso sobre pelvis y tobillos. La amplitud de avance (x)
        escala la zancada; la de giro (a) introduce una asimetría de cadera
        entre ambas piernas para rotar.
        """
        if phase is None:
            phase = self.phase

        pose = dict(config.STANDING_POSE)

        x = self.x_amplitude
        a = self.a_amplitude
        hip = config.GAIT_HIP_AMPLITUDE
        knee = config.GAIT_KNEE_AMPLITUDE
        ankle = config.GAIT_ANKLE_AMPLITUDE
        sway = config.GAIT_LATERAL_SWAY
        offset = config.GAIT_LEG_PHASE_OFFSET

        # Fases de cada pierna (180 grados de desfase).
        phase_r = phase
        phase_l = phase + offset

        sin_r = math.sin(phase_r)
        sin_l = math.sin(phase_l)
        # La rodilla flexiona en el doble de frecuencia (levanta el pie en la
        # fase de balanceo): usamos |sin| para que siempre flexione, no estire.
        knee_r = abs(math.sin(phase_r))
        knee_l = abs(math.sin(phase_l))
        # Balanceo lateral: un coseno a la frecuencia del ciclo desplaza el peso.
        lateral = math.cos(phase)

        # Avance (x): oscilación de cadera + flexión de rodilla + tobillo.
        pose["LegUpperR"] += x * hip * sin_r
        pose["LegUpperL"] += x * hip * sin_l
        pose["LegLowerR"] += x * knee * knee_r
        pose["LegLowerL"] += -x * knee * knee_l
        pose["AnkleR"] += x * ankle * sin_r
        pose["AnkleL"] += x * ankle * sin_l

        # Giro (a): asimetría de cadera-yaw entre piernas (a>0 gira a izquierda).
        pose["PelvYR"] += a * 0.3
        pose["PelvYL"] += a * 0.3

        # Balanceo lateral del peso (sway) sobre pelvis y tobillos.
        pose["PelvR"] += sway * lateral
        pose["PelvL"] += sway * lateral
        pose["AnkleR"] += -sway * lateral
        pose["AnkleL"] += -sway * lateral

        return {name: _clamp_joint(name, angle) for name, angle in pose.items()}


class _LocomotionBackend:
    """Locomoción por control directo de motores con gait sinusoidal propio.

    Los managers de Robotis (RobotisOp2GaitManager/MotionManager) existen solo
    como binario C++; `import managers` lanza ModuleNotFoundError en Python. No
    hay vía oficial Python para caminar, así que accionamos los 20 motores del
    DARwIn-OP directamente con setPosition cada paso, usando `_GaitEngine`.

    En Webots real: `getMotor`/`getDevice` resuelven los 20 motores y el gait
    los mueve. En el mock headless (que no tiene PositionSensors ni física de
    marcha) el avance lo sigue produciendo `queue_action` + `_apply_kinematics`,
    por lo que cada acción conserva su llamada a `_queue(...)`.
    """

    def __init__(self, supervisor, config_path: str = ""):
        self.supervisor = supervisor
        self.engine = _GaitEngine()
        self.motors: Dict[str, object] = {}
        self._fetch_motors()
        # available = hay motores reales que accionar (Webots). Si ninguno
        # resolvió, operamos en modo no-op (mock sin motores) sin romper tests.
        self.available = bool(self.motors)
        if not self.available:
            warnings.warn(
                "Ningún motor del DARwIn-OP resolvió vía getMotor/getDevice; "
                "locomoción en modo no-op (backend sin motores).",
                RuntimeWarning,
                stacklevel=2,
            )

    def _fetch_motors(self) -> None:
        getter = getattr(self.supervisor, "getMotor", None)
        if getter is None:
            getter = getattr(self.supervisor, "getDevice", None)
        if getter is None:
            return
        for name in config.MOTOR_NAMES:
            try:
                motor = getter(name)
            except Exception:
                motor = None
            if motor is None:
                continue
            # Fijar velocidad por defecto si el motor lo soporta.
            set_vel = getattr(motor, "setVelocity", None)
            if set_vel is not None:
                try:
                    set_vel(config.MOTOR_VELOCITY)
                except Exception:
                    pass
            self.motors[name] = motor

    # --- acciones: fijan las amplitudes de comando del gait -----------------
    def walk_forward(self) -> None:
        self.engine.set_amplitudes(config.GAIT_X_AMPLITUDE, 0.0)
        _queue(self.supervisor, "walk_forward")

    def turn_left(self) -> None:
        self.engine.set_amplitudes(0.0, config.GAIT_A_AMPLITUDE)
        _queue(self.supervisor, "turn_left")

    def turn_right(self) -> None:
        self.engine.set_amplitudes(0.0, -config.GAIT_A_AMPLITUDE)
        _queue(self.supervisor, "turn_right")

    def stand(self) -> None:
        self.engine.set_amplitudes(0.0, 0.0)
        _queue(self.supervisor, None)

    # --- accionamiento de motores ------------------------------------------
    def apply_targets(self, targets: Dict[str, float]) -> None:
        """Comanda setPosition a cada motor resuelto con los ángulos dados."""
        for name, motor in self.motors.items():
            angle = targets.get(name)
            if angle is None:
                continue
            set_pos = getattr(motor, "setPosition", None)
            if set_pos is None:
                continue
            try:
                set_pos(float(angle))
            except Exception:
                pass

    def update(self) -> None:
        """Un paso de gait: avanza la fase y comanda los motores."""
        self.engine.advance()
        self.apply_targets(self.engine.compute_targets())

    def hold_standing(self) -> None:
        """Comanda la pose de pie sin avanzar la fase (estabilización)."""
        self.engine.set_amplitudes(0.0, 0.0)
        self.apply_targets(self.engine.standing_pose())


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
        """Actualiza el gait (setPosition a los motores) y avanza un paso.

        El gait se recalcula CADA paso de simulación: primero se comandan los
        motores según la fase actual, luego se avanza la física del backend.
        """
        self.locomotion.update()
        return self.backend.step(self.time_step)

    def stabilize(self) -> None:
        """Mantiene la pose de pie durante STABILIZATION_STEPS ticks quietos.

        Sirve para que la física del robot asiente antes de empezar el episodio.
        No avanza la fase del gait (pose estática). Es barato bajo el mock.
        """
        self.locomotion.engine.reset_phase()
        for _ in range(int(config.STABILIZATION_STEPS)):
            self.locomotion.hold_standing()
            self.backend.step(self.time_step)

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
        """True si el robot está caído, combinando tres señales independientes.

        (a) Acelerómetro: en Webots el DARwIn-OP lee cada eje centrado en
            ~ACCEL_CENTER (512) estando de pie. Una desviación mayor que
            ACCEL_FALL_TOLERANCE indica inclinación/caída.
        (b) Orientación del torso vía Supervisor: el ángulo del eje-ángulo del
            campo 'rotation' por encima de FALL_TILT_THRESHOLD indica vuelco.
        (c) Altura: la z del torso por debajo de FALL_HEIGHT_Z indica colapso.

        Cada rama se evalúa defensivamente: si falta el sensor o el campo, se
        omite esa señal sin lanzar excepción. Caído = cualquiera de ellas.
        """
        # (a) Acelerómetro centrado en ~512.
        if self._accel is not None:
            try:
                values = self._accel.getValues()
            except Exception:
                values = None
            if values is not None:
                deviation = max(abs(float(v) - config.ACCEL_CENTER) for v in values)
                if deviation > config.ACCEL_FALL_TOLERANCE:
                    return True

        # (b) Inclinación del torso desde el campo 'rotation' del Supervisor.
        try:
            rot = self._self_node.getField("rotation").getSFRotation()
            tilt_angle = abs(float(rot[3]))
            # Normalizar a [0, pi]: un giro de yaw puro (eje z) no es caída.
            axis_x, axis_y, axis_z = float(rot[0]), float(rot[1]), float(rot[2])
            horizontal_axis = math.hypot(axis_x, axis_y)
            # Inclinación efectiva = ángulo proyectado sobre el eje horizontal.
            effective_tilt = tilt_angle * horizontal_axis
            if effective_tilt > config.FALL_TILT_THRESHOLD:
                return True
        except Exception:
            pass

        # (c) Altura del torso.
        try:
            trans = self._self_node.getField("translation").getSFVec3f()
            if float(trans[2]) < config.FALL_HEIGHT_Z:
                return True
        except Exception:
            pass

        return False

    # --- pose ---------------------------------------------------------------
    def reset_pose(self, x: float, y: float, z: Optional[float] = None) -> None:
        """Prepara el robot para un nuevo episodio.

        Pasos: reubica en (x, y[, z]) mirando a la meta (heading 0) —
        reposicionar al inicio es legítimo; resetea la fase del gait a 0 y
        comanda la pose de pie; reinicia la física del nodo propio con
        resetPhysics() cuando está disponible (el mock no lo tiene, se omite);
        y corre la estabilización (STABILIZATION_STEPS ticks quietos de pie).

        NO se falsea avance mutando `translation` durante step(): el progreso
        debe emerger del gait.
        """
        trans_field = self._self_node.getField("translation")
        current = trans_field.getSFVec3f()
        new_z = current[2] if z is None else z
        trans_field.setSFVec3f([float(x), float(y), float(new_z)])
        self._self_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])

        # Pose de pie + fase de gait a cero.
        self.locomotion.engine.reset_phase()
        self.locomotion.hold_standing()

        # Reinicio de la física del nodo propio (solo Webots real).
        reset_physics = getattr(self._self_node, "resetPhysics", None)
        if reset_physics is not None:
            try:
                reset_physics()
            except Exception:
                pass

        # Periodo de estabilización antes de comenzar el episodio.
        self.stabilize()

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
