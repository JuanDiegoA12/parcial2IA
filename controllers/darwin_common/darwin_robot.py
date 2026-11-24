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


def _iter_world_nodes(supervisor):
    """Itera los nodos hijos de la raíz de la escena del supervisor.

    Compartido por DarwinRobot (find_others/find_barriers) y por
    _LocomotionBackend (bloqueo de colisión cinemático), de modo que ambos
    escanean la escena de la MISMA forma. Defensivo: si falta la raíz o los
    hijos, no itera nada.
    """
    get_root = getattr(supervisor, "getRoot", None)
    if get_root is None:
        return
    root = get_root()
    if root is None:
        return
    try:
        children = root.getField("children")
        count = children.getCount()
    except Exception:
        return
    for i in range(count):
        try:
            node = children.getMFNode(i)
        except Exception:
            node = None
        if node is None:
            continue
        yield node


def _node_name(node) -> Optional[str]:
    """Nombre de un nodo de escena (getNameString o campo 'name'). None si falla."""
    getter = getattr(node, "getNameString", None)
    if getter is not None:
        try:
            return getter()
        except Exception:
            pass
    try:
        name_field = node.getField("name")
    except Exception:
        return None
    if name_field is None:
        return None
    try:
        return name_field.getSFString()
    except Exception:
        value = getattr(name_field, "_value", None)
        if isinstance(value, (list, tuple)) and value:
            return value[0]
    return None


def _node_xy(node) -> Optional[Tuple[float, float]]:
    """(x, y) del campo translation de un nodo. None si no se puede leer."""
    try:
        trans = node.getField("translation").getSFVec3f()
    except Exception:
        return None
    return float(trans[0]), float(trans[1])


def _clamp_to_limits(limits: Optional[Tuple[float, float]], angle: float) -> float:
    """Recorta `angle` a [min+margen, max-margen] de `limits` (o lo deja igual).

    Si `limits` es None el ángulo se devuelve intacto (motor sin límites
    conocidos). Nunca se comanda una articulación exactamente a su tope físico:
    así se evitan las advertencias "too big/low requested position" de Webots.
    Si el margen excede el rango, se devuelve el centro del rango.
    """
    if limits is None:
        return float(angle)
    lo, hi = limits
    margin = config.JOINT_LIMIT_MARGIN
    lo_safe = lo + margin
    hi_safe = hi - margin
    if lo_safe > hi_safe:  # margen mayor que el rango: usa el centro
        return float((lo + hi) / 2.0)
    return float(max(lo_safe, min(hi_safe, angle)))


def _clamp_joint(name: str, angle: float) -> float:
    """Recorta `angle` contra el fallback de config.JOINT_LIMITS del motor.

    Es el recorte por defecto de nivel módulo que usan `_GaitEngine` y los
    tests puros de config. En Webots real el recorte autoritativo lo aplica
    `_LocomotionBackend.clamp`, que usa los límites leídos en runtime.
    """
    return _clamp_to_limits(config.JOINT_LIMITS.get(name), angle)


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
        # Modo de locomoción resuelto una sola vez al construir: 'gait' (Plan A,
        # gait sinusoidal físico) o 'kinematic' (Plan B, arrastre vía Supervisor).
        self.mode = config.resolve_locomotion_mode()
        # Acción cinemática pendiente para aplicar en el próximo update() en
        # modo kinematic (None, 'walk_forward', 'turn_left' o 'turn_right').
        self._kin_action: Optional[str] = None
        self.motors: Dict[str, object] = {}
        # Límites REALES por motor leídos en runtime (ver _fetch_motors). Fuente
        # autoritativa del recorte en Webots; vacío bajo el mock sin motores.
        self.motor_limits: Dict[str, Tuple[float, float]] = {}
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
            # Leer los límites REALES del motor en runtime. Si el motor no
            # expone los getters (p.ej. mock antiguo), caer al fallback de
            # config.JOINT_LIMITS; si tampoco está, dejar el motor sin límites.
            self.motor_limits[name] = self._read_motor_limits(name, motor)

    @staticmethod
    def _read_motor_limits(name: str, motor) -> Optional[Tuple[float, float]]:
        """Lee (min, max) reales del motor, con fallback a config.JOINT_LIMITS.

        Guardado defensivamente: el mock puede carecer de
        getMinPosition/getMaxPosition. Devuelve None si no hay ningún límite
        conocido (motor sin cota).
        """
        lo = hi = None
        get_min = getattr(motor, "getMinPosition", None)
        get_max = getattr(motor, "getMaxPosition", None)
        if callable(get_min) and callable(get_max):
            try:
                lo = float(get_min())
                hi = float(get_max())
            except Exception:
                lo = hi = None
        if lo is None or hi is None or lo >= hi:
            return config.JOINT_LIMITS.get(name)
        return (lo, hi)

    def clamp(self, name: str, angle: float) -> float:
        """Recorta `angle` contra los límites REALES del motor (o el fallback).

        Usa los límites leídos en runtime en `self.motor_limits`; si el nombre
        no está, cae al recorte de nivel módulo contra config.JOINT_LIMITS.
        """
        if name in self.motor_limits:
            return _clamp_to_limits(self.motor_limits[name], angle)
        return _clamp_joint(name, angle)

    # --- acciones: fijan las amplitudes de comando del gait -----------------
    # En modo kinematic además se registra la acción pendiente en _kin_action
    # para que update() la traduzca a un arrastre por campos del Supervisor. Las
    # líneas de gait (set_amplitudes + _queue) quedan INTACTAS en ambos modos.
    def walk_forward(self) -> None:
        # Las líneas del gait (set_amplitudes) quedan INTACTAS en ambos modos.
        self.engine.set_amplitudes(config.GAIT_X_AMPLITUDE, 0.0)
        # En modo kinematic, BLOQUEAR el avance si el destino penetraría una
        # valla u otro robot: no se encola la acción (ni el mock ni el camino
        # real la aplican) y _kin_action queda en None para este tick, de modo
        # que el robot no atraviesa el obstáculo. El giro nunca se bloquea, así
        # el agente puede reorientarse y esquivar. En modo gait no hay bloqueo.
        if self.mode == "kinematic":
            if self._kinematic_blocked_forward():
                self._kin_action = None
                return
            _queue(self.supervisor, "walk_forward")
            self._kin_action = "walk_forward"
            return
        _queue(self.supervisor, "walk_forward")

    def turn_left(self) -> None:
        self.engine.set_amplitudes(0.0, config.GAIT_A_AMPLITUDE)
        _queue(self.supervisor, "turn_left")
        if self.mode == "kinematic":
            self._kin_action = "turn_left"

    def turn_right(self) -> None:
        self.engine.set_amplitudes(0.0, -config.GAIT_A_AMPLITUDE)
        _queue(self.supervisor, "turn_right")
        if self.mode == "kinematic":
            self._kin_action = "turn_right"

    def stand(self) -> None:
        self.engine.set_amplitudes(0.0, 0.0)
        _queue(self.supervisor, None)
        if self.mode == "kinematic":
            self._kin_action = None

    # --- accionamiento de motores ------------------------------------------
    def apply_targets(self, targets: Dict[str, float]) -> None:
        """Comanda setPosition a cada motor resuelto con los ángulos dados.

        Cada ángulo pasa por `self.clamp` como guarda final contra los límites
        REALES del motor (leídos en runtime), de modo que en Webots ganan los
        topes verdaderos y nunca se emite "too big/low requested position".
        """
        for name, motor in self.motors.items():
            angle = targets.get(name)
            if angle is None:
                continue
            set_pos = getattr(motor, "setPosition", None)
            if set_pos is None:
                continue
            try:
                set_pos(self.clamp(name, float(angle)))
            except Exception:
                pass

    def _apply_kinematic_motion(self, action: Optional[str]) -> None:
        """Arrastre cinemático: escribe translation/rotation del nodo propio.

        SÓLO actúa en Webots real, detectado porque el backend NO expone
        `queue_action` (el mock sí lo expone y ya produce el movimiento vía
        `_apply_kinematics` con la MISMA convención de signos/ejes, así que si
        también escribiéramos aquí el desplazamiento se aplicaría DOS veces).
        La convención coincide con la del mock (tests/conftest.py):
        heading 0 mira a +y; avance = (-sin(theta)*d, +cos(theta)*d);
        girar suma +/- el paso angular al ángulo de rotación en z;
        theta = rot[3] si rot[2] >= 0, si no -rot[3].

        Pin visual de pie (aproximación): al avanzar, la z se fija a
        config.KINEMATIC_STAND_Z (no se hereda la z colapsada) y la rotación se
        mantiene como yaw puro sobre z (roll/pitch = 0). El usuario debe
        verificar en Webots que el torso no se acuesta a esa z.
        """
        if action is None:
            return
        # Bajo el mock el movimiento lo produce queue_action + _apply_kinematics.
        if hasattr(self.supervisor, "queue_action"):
            return
        get_self = getattr(self.supervisor, "getSelf", None)
        if get_self is None:
            return
        node = get_self()
        if node is None:
            return
        try:
            trans_field = node.getField("translation")
            rot_field = node.getField("rotation")
            rot = rot_field.getSFRotation()
            theta = rot[3] if rot[2] >= 0 else -rot[3]
            if action == "walk_forward":
                d = config.KINEMATIC_LINEAR_STEP
                x, y, _z = trans_field.getSFVec3f()
                x += -math.sin(theta) * d
                y += math.cos(theta) * d
                # Pin visual de pie: fija z a KINEMATIC_STAND_Z en vez de heredar
                # la z (posiblemente colapsada), para que el robot se deslice DE
                # PIE. Es una APROXIMACIÓN visual: la rotación ya es yaw puro
                # ([0,0,1,theta], sin roll/pitch) y no se toca aquí. El usuario
                # debe verificar en Webots que el torso no se acuesta.
                trans_field.setSFVec3f(
                    [float(x), float(y), float(config.KINEMATIC_STAND_Z)]
                )
            elif action == "turn_left":
                theta += config.KINEMATIC_ANGULAR_STEP
                rot_field.setSFRotation([0.0, 0.0, 1.0, float(theta)])
            elif action == "turn_right":
                theta -= config.KINEMATIC_ANGULAR_STEP
                rot_field.setSFRotation([0.0, 0.0, 1.0, float(theta)])
        except Exception:
            pass

    def _self_xy_theta(self) -> Optional[Tuple[float, float, float]]:
        """(x, y, theta) del nodo propio, con la MISMA convención de heading que
        _apply_kinematic_motion (theta = rot[3] si rot[2] >= 0, si no -rot[3]).

        Devuelve None si no hay nodo propio resoluble (p.ej. robot no Supervisor),
        de modo que el bloqueo se desactive de forma segura en ese caso.
        """
        get_self = getattr(self.supervisor, "getSelf", None)
        if get_self is None:
            return None
        try:
            node = get_self()
        except Exception:
            return None
        if node is None:
            return None
        try:
            x, y, _z = node.getField("translation").getSFVec3f()
            rot = node.getField("rotation").getSFRotation()
        except Exception:
            return None
        theta = rot[3] if rot[2] >= 0 else -rot[3]
        return float(x), float(y), float(theta)

    def _obstacles_xy(self):
        """(x, y) de las vallas y de los OTROS robots (excluye el nodo propio).

        Escanea la escena con los mismos helpers que DarwinRobot.find_barriers /
        find_others, para que el bloqueo use exactamente la misma fuente. El nodo
        propio se excluye por identidad (no por nombre) para no auto-bloquearse.
        """
        get_self = getattr(self.supervisor, "getSelf", None)
        self_node = None
        if get_self is not None:
            try:
                self_node = get_self()
            except Exception:
                self_node = None
        robot_names = set(config.START_POSITIONS.keys())
        points = []
        for node in _iter_world_nodes(self.supervisor):
            if self_node is not None and node is self_node:
                continue
            name = _node_name(node)
            if name is None:
                continue
            is_barrier = name.startswith("valla")
            is_other_robot = name in robot_names
            if not (is_barrier or is_other_robot):
                continue
            xy = _node_xy(node)
            if xy is not None:
                points.append(xy)
        return points

    def _kinematic_blocked_forward(self) -> bool:
        """True si un paso de walk_forward penetraría una valla u otro robot.

        Calcula el destino con la MISMA fórmula de avance del camino cinemático
        (-sin(theta)*d, +cos(theta)*d) y devuelve True si ese destino cae dentro
        de KINEMATIC_COLLISION_RADIUS de algún obstáculo. El giro NO se consulta
        aquí (nunca se bloquea), así el agente puede reorientarse y esquivar.
        Defensivo: si no hay nodo propio o la escena no se puede leer, no bloquea.
        """
        pose = self._self_xy_theta()
        if pose is None:
            return False
        x, y, theta = pose
        d = config.KINEMATIC_LINEAR_STEP
        nx = x - math.sin(theta) * d
        ny = y + math.cos(theta) * d
        radius = config.KINEMATIC_COLLISION_RADIUS
        for (bx, by) in self._obstacles_xy():
            if math.hypot(nx - bx, ny - by) < radius:
                return True
        return False

    def update(self) -> None:
        """Un paso de locomoción. En gait (Plan A) avanza la fase y comanda los
        motores; en kinematic (Plan B) mantiene la pose de pie, aplica el
        arrastre por campos del Supervisor y reinicia la física para no
        acumular caída/deriva."""
        if self.mode == "kinematic":
            # Mantener la pose de pie en los motores (no-op si no hay motores,
            # p.ej. el mock), arrastrar al robot y reiniciar su física.
            self.hold_standing()
            self._apply_kinematic_motion(self._kin_action)
            self._reset_self_physics()
            self._kin_action = None
            return
        self.engine.advance()
        self.apply_targets(self.engine.compute_targets())

    def _reset_self_physics(self) -> None:
        """Reinicia la física del nodo propio en Webots (no-op bajo el mock).

        Evita que el teletransporte por paso acumule velocidad/caída. Si el
        nodo no expone resetPhysics (mock), se omite silenciosamente.
        """
        get_self = getattr(self.supervisor, "getSelf", None)
        if get_self is None:
            return
        node = get_self()
        reset_physics = getattr(node, "resetPhysics", None)
        if reset_physics is None:
            return
        try:
            reset_physics()
        except Exception:
            pass

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
        # getSelf() devuelve el nodo propio SÓLO si el robot es Supervisor
        # ('supervisor TRUE' en el .wbt). Si no lo es, Webots registra
        # "illegal call to wb_supervisor_node_get_self() in a 'Robot'
        # controller" y getSelf() devuelve None, lo que reventaría más tarde
        # con AttributeError al pedir getField(...). Guardamos la llamada y
        # dejamos self._self_node = None para que is_supervisor() lo detecte y
        # el controlador avise con un mensaje accionable en vez de reventar.
        try:
            self._self_node = backend.getSelf()
        except Exception:
            self._self_node = None
        self._accel = self._safe_sensor("getAccelerometer", "Accelerometer")
        self._gyro = self._safe_sensor("getGyro", "Gyro")
        # Contador de ticks consecutivos con el acelerómetro fuera de banda.
        # Sólo tras FALL_ACCEL_PERSIST_STEPS seguidos se declara caída por
        # acelerómetro (persistencia tipo Walk.cpp); un pico aislado se ignora.
        self._accel_out_of_band = 0

    # --- supervisor ---------------------------------------------------------
    def is_supervisor(self) -> bool:
        """True si el robot resolvió su nodo propio vía getSelf().

        Es False cuando el nodo del robot NO tiene 'supervisor TRUE' en el
        .wbt: en ese caso getSelf() devuelve None (o lanza) y no se puede leer
        posición/orientación ni reubicar. Los controladores deben comprobarlo
        y avisar con un mensaje accionable antes de usar el robot.
        """
        return self._self_node is not None

    SUPERVISOR_REQUIRED_MSG = (
        "Este robot no es Supervisor: getSelf() devolvió None. Añade "
        "'supervisor TRUE' al nodo del robot en el .wbt para usar "
        "locomotion_test / el modo kinemático."
    )

    def _require_self_node(self):
        """Devuelve el nodo propio o lanza RuntimeError con mensaje accionable."""
        if self._self_node is None:
            raise RuntimeError(self.SUPERVISOR_REQUIRED_MSG)
        return self._self_node

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
        trans = self._require_self_node().getField("translation").getSFVec3f()
        return float(trans[0]), float(trans[1]), float(trans[2])

    def get_orientation(self) -> float:
        """Heading (rad) alrededor de z. 0 = mirando hacia +y (eje de carrera)."""
        rot = self._require_self_node().getField("rotation").getSFRotation()
        axis_z = rot[2]
        angle = rot[3]
        return float(angle if axis_z >= 0 else -angle)

    def is_fallen(self) -> bool:
        """True si el robot está caído, combinando tres señales independientes.

        (c) Altura (PRIMARIA): la z del torso por debajo de FALL_HEIGHT_Z indica
            colapso; es una señal inmediata.
        (b) Orientación del torso vía Supervisor: el ángulo del eje-ángulo del
            campo 'rotation' (ignorando el yaw puro en z) por encima de
            FALL_TILT_THRESHOLD indica vuelco.
        (a) Acelerómetro: en Webots el DARwIn-OP lee cada eje centrado en
            ~ACCEL_CENTER (512) estando de pie. La lectura es RUIDOSA, así que
            una desviación puntual mayor que ACCEL_FALL_TOLERANCE NO basta:
            hace falta PERSISTENCIA de FALL_ACCEL_PERSIST_STEPS ticks
            consecutivos fuera de banda (igual que Walk.cpp). Un pico aislado
            incrementa el contador pero no declara caída; una lectura dentro de
            banda lo reinicia a 0.

        Cada rama se evalúa defensivamente: si falta el sensor o el campo, se
        omite esa señal sin lanzar excepción. Caído = cualquiera de ellas.

        En modo kinematic (Plan B) la caída está DESACTIVADA por diseño: el
        robot se desliza de pie vía Supervisor y no debe volcar, así que
        is_fallen() devuelve siempre False (las penalizaciones por
        proximidad/colisión y el cruce de meta siguen activas en race_env).
        """
        if self.locomotion.mode == "kinematic":
            return False

        # (a) Acelerómetro centrado en ~512 CON PERSISTENCIA. Se evalúa primero
        # para mantener el contador actualizado cada tick, pero sólo declara
        # caída tras FALL_ACCEL_PERSIST_STEPS ticks seguidos fuera de banda.
        if self._accel is not None:
            try:
                values = self._accel.getValues()
            except Exception:
                values = None
            if values is not None:
                deviation = max(abs(float(v) - config.ACCEL_CENTER) for v in values)
                if deviation > config.ACCEL_FALL_TOLERANCE:
                    self._accel_out_of_band += 1
                    if self._accel_out_of_band >= config.FALL_ACCEL_PERSIST_STEPS:
                        return True
                else:
                    self._accel_out_of_band = 0

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
        self_node = self._require_self_node()
        trans_field = self_node.getField("translation")
        current = trans_field.getSFVec3f()
        new_z = current[2] if z is None else z
        trans_field.setSFVec3f([float(x), float(y), float(new_z)])
        self_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])

        # Pose de pie + fase de gait a cero. También se limpia cualquier acción
        # cinemática pendiente para que el modo kinematic arranque el episodio
        # sin arrastre residual.
        self.locomotion.engine.reset_phase()
        self.locomotion._kin_action = None
        self.locomotion.hold_standing()

        # Reinicio de la física del nodo propio (solo Webots real).
        reset_physics = getattr(self_node, "resetPhysics", None)
        if reset_physics is not None:
            try:
                reset_physics()
            except Exception:
                pass

        # Periodo de estabilización antes de comenzar el episodio.
        self.stabilize()

    # --- escena -------------------------------------------------------------
    def find_others(self) -> Dict[str, Tuple[float, float]]:
        """Otros robots por nombre (excluye self). Nunca usa getFromDef."""
        robot_names = set(config.START_POSITIONS.keys())
        others: Dict[str, Tuple[float, float]] = {}
        for node in _iter_world_nodes(self.backend):
            name = _node_name(node)
            if name is None or name == self.name or name not in robot_names:
                continue
            xy = _node_xy(node)
            if xy is not None:
                others[name] = xy
        return others

    def find_barriers(self) -> Dict[str, Tuple[float, float]]:
        """Vallas/obstáculos por nombre (los nodos cuyo nombre empieza por 'valla')."""
        barriers: Dict[str, Tuple[float, float]] = {}
        for node in _iter_world_nodes(self.backend):
            name = _node_name(node)
            if name is None or not name.startswith("valla"):
                continue
            xy = _node_xy(node)
            if xy is not None:
                barriers[name] = xy
        return barriers
