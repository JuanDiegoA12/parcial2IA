"""Mock en memoria de la API de Webots para ejecutar los tests sin simulador.

Reproduce lo mínimo de `controller` (Supervisor/Robot, Node, Field,
Accelerometer, Gyro, Motor) y del módulo compilado `managers`
(RobotisOp2GaitManager, RobotisOp2MotionManager) que usa la capa de robot.

El mock mantiene N nodos de robot nombrados (DQN, ppo, actor - critic,
agente propio) y las 12 vallas, cada uno con translation/rotation mutables, y
un step() que avanza un modelo cinemático de juguete:

- walk_forward: avanza +delta a lo largo del heading actual.
- turn_left / turn_right: cambian el heading.
- una inclinación (tilt) forzable simula una caída.

La convención de heading: el ángulo `theta` es la rotación alrededor de z. A
theta=0 el robot mira hacia +y (eje de la carrera), de modo que walk_forward
acerca el robot a la meta. Vector de avance = (-sin theta, cos theta).
"""

from __future__ import annotations

import math
import sys
import types
from typing import Dict, List

import pytest

# Paso cinemático de juguete por acción (en unidades del mundo / radianes).
WALK_DELTA = 0.25   # metros por walk_forward
TURN_DELTA = 0.20   # radianes por turn_left/turn_right

# Posiciones iniciales de los robots (deben coincidir con config/START_POSITIONS).
_ROBOT_STARTS = {
    "DQN": (0.11, 9.95, 0.24),
    "ppo": (0.21, 7.88, 0.34),
    "actor - critic": (0.21, 6.04, 0.34),
    "agente propio": (0.21, 11.99, 0.34),
}

# Las 12 vallas tomadas del .wbt (name -> translation).
_BARRIERS = {
    "valla 1": (12.685, -0.114, 0.0),
    "valla 1(1)": (10.615, -0.114002, 0.0),
    "valla 1(4)": (-10.415, -0.114068, 0.0),
    "valla 1(8)": (-0.895, -7.97407, 0.0),
    "valla 1(9)": (-0.895005, -5.97407, 0.0),
    "valla 1(10)": (-0.89499, -9.9841, 0.0),
    "valla 1(11)": (-0.894983, -12.0141, 0.0),
    "valla 1(5)": (-12.655, -0.114071, 0.0),
    "valla 1(6)": (-14.635, -0.114071, 0.0),
    "valla 1(7)": (-16.695, -0.114071, 0.0),
    "valla 1(2)": (14.645, -0.113995, 0.0),
    "valla 1(3)": (16.685, -0.113996, 0.0),
}


class FakeField:
    """Campo SF* de un nodo (translation o rotation)."""

    def __init__(self, value: List[float]):
        self._value = list(value)

    def getSFVec3f(self) -> List[float]:
        return list(self._value)

    def setSFVec3f(self, value) -> None:
        self._value = list(value)

    def getSFRotation(self) -> List[float]:
        return list(self._value)

    def setSFRotation(self, value) -> None:
        self._value = list(value)


class FakeNode:
    """Nodo del árbol de la escena con nombre, translation y rotation."""

    def __init__(self, name: str, translation, rotation=(0.0, 0.0, 1.0, 0.0)):
        self.name = name
        self._fields = {
            "name": FakeField([name]),  # no se usa como Vec3f; placeholder
            "translation": FakeField(list(translation)),
            "rotation": FakeField(list(rotation)),
        }

    def getField(self, name: str) -> FakeField:
        return self._fields[name]

    # Helpers de conveniencia para el modelo cinemático y las aserciones.
    @property
    def translation(self) -> List[float]:
        return self._fields["translation"].getSFVec3f()

    @property
    def rotation(self) -> List[float]:
        return self._fields["rotation"].getSFRotation()

    def set_translation(self, value) -> None:
        self._fields["translation"].setSFVec3f(value)

    def set_rotation(self, value) -> None:
        self._fields["rotation"].setSFRotation(value)

    def getNameString(self) -> str:
        return self.name


class FakeRootField:
    """Campo 'children' del nodo raíz: lista MF de nodos."""

    def __init__(self, nodes: List[FakeNode]):
        self._nodes = nodes

    def getCount(self) -> int:
        return len(self._nodes)

    def getMFNode(self, index: int) -> FakeNode:
        return self._nodes[index]


class FakeRoot:
    def __init__(self, nodes: List[FakeNode]):
        self._children = FakeRootField(nodes)

    def getField(self, name: str) -> FakeRootField:
        assert name == "children"
        return self._children


class FakeSensor:
    """Acelerómetro / giroscopio con valores forzables.

    Modela el acelerómetro del DARwIn-OP de Webots: de pie cada eje lee
    centrado en ~512 (no un modelo (0,0,9.81)). Por eso el valor por defecto es
    (512, 512, 512); una caída se fuerza con FakeSupervisor.force_tilt.
    """

    def __init__(self, values=(512.0, 512.0, 512.0)):
        self._values = list(values)

    def enable(self, sampling_period: int) -> None:
        pass

    def getValues(self) -> List[float]:
        return list(self._values)

    def set_values(self, values) -> None:
        self._values = list(values)


class FakeMotor:
    def __init__(self, name: str):
        self.name = name
        self.position = 0.0

    def setPosition(self, position: float) -> None:
        self.position = position

    def setVelocity(self, velocity: float) -> None:
        self.velocity = velocity

    def enableTorqueFeedback(self, period: int) -> None:
        pass


class FakeSupervisor:
    """Mock de controller.Supervisor compartido por todos los robots.

    Mantiene el estado global del mundo (todos los robots + vallas). Cada
    instancia representa un robot concreto (`self_name`) que obtiene su propio
    nodo vía getSelf(). step() aplica la acción pendiente del robot self.
    """

    # Estado del mundo compartido entre instancias (como en Webots real, donde
    # todos ven el mismo árbol de escena). Se reinicia por fixture.
    _world: Dict[str, FakeNode] = {}
    _root: FakeRoot = None

    def __init__(self, self_name: str = "DQN", time_step: int = 32):
        self.self_name = self_name
        self.time_step = time_step
        self._accel = FakeSensor()
        self._gyro = FakeSensor(values=(0.0, 0.0, 0.0))
        self._motors: Dict[str, FakeMotor] = {}
        # Acción pendiente aplicada en el próximo step(): 'walk_forward',
        # 'turn_left', 'turn_right' o None.
        self._pending_action = None

    # --- API de Webots ------------------------------------------------------
    def getBasicTimeStep(self) -> float:
        return float(self.time_step)

    def getSelf(self) -> FakeNode:
        return self._world[self.self_name]

    def getFromDef(self, name: str):
        # Todos los robots comparten DEF 'DARWIN'; devolver el primero que
        # encaje emula la ambigüedad real (no debe usarse para distinguir).
        for node in self._world.values():
            return node
        return None

    def getRoot(self) -> FakeRoot:
        return self._root

    def getAccelerometer(self, name: str) -> FakeSensor:
        return self._accel

    def getGyro(self, name: str) -> FakeSensor:
        return self._gyro

    def getDevice(self, name: str):
        # Sensores por nombre o motores.
        if name in ("Accelerometer", "accelerometer"):
            return self._accel
        if name in ("Gyro", "gyro"):
            return self._gyro
        if name not in self._motors:
            self._motors[name] = FakeMotor(name)
        return self._motors[name]

    def getMotor(self, name: str) -> FakeMotor:
        if name not in self._motors:
            self._motors[name] = FakeMotor(name)
        return self._motors[name]

    def step(self, time_step: int = None) -> int:
        self._apply_kinematics()
        return 0

    # --- Modelo cinemático de juguete --------------------------------------
    def queue_action(self, action: str) -> None:
        self._pending_action = action

    def force_tilt(self, tilt_rad: float) -> None:
        """Interruptor único de "forzar caída", coherente con is_fallen().

        `is_fallen()` combina tres señales (acelerómetro centrado ~512,
        orientación del torso y altura). Para que un valor grande de
        `tilt_rad` sea inequívocamente una caída y uno pequeño no lo sea,
        `force_tilt` ajusta las TRES de forma coherente:

        - Acelerómetro: desvía los ejes de 512 en proporción a tilt_rad (una
          inclinación grande saca la lectura de la banda tolerada).
        - Orientación: fija una rotación de `tilt_rad` alrededor de un eje
          horizontal (x) para que el torso aparezca volcado.
        - Altura: baja la z del torso cuando la inclinación es grande.

        Con tilt_rad pequeño (p.ej. 0.1) las tres señales quedan dentro de lo
        normal -> no caído.
        """
        # (a) Acelerómetro: desviación proporcional a la inclinación. Con
        # tilt grande la desviación supera ACCEL_FALL_TOLERANCE (~80).
        deviation = math.sin(tilt_rad) * 512.0
        self._accel.set_values((512.0 + deviation, 512.0, 512.0 - deviation))

        # (b) Orientación: rotación alrededor del eje x horizontal (vuelco).
        node = self._world[self.self_name]
        node.set_rotation([1.0, 0.0, 0.0, float(tilt_rad)])

        # (c) Altura: baja el torso cuando la inclinación es grande; con
        # inclinación pequeña lo restaura a una altura de pie segura (coherente
        # con "no caído"), de modo que force_tilt sea reversible.
        x, y, _z = node.translation
        new_z = 0.1 if tilt_rad > 0.5 else 0.3
        node.set_translation([x, y, new_z])

    def _apply_kinematics(self) -> None:
        action = self._pending_action
        self._pending_action = None
        if action is None:
            return
        node = self._world[self.self_name]
        rot = node.rotation
        theta = rot[3] if rot[2] >= 0 else -rot[3]
        if action == "walk_forward":
            x, y, z = node.translation
            x += -math.sin(theta) * WALK_DELTA
            y += math.cos(theta) * WALK_DELTA
            node.set_translation([x, y, z])
        elif action == "turn_left":
            theta += TURN_DELTA
            node.set_rotation([0.0, 0.0, 1.0, theta])
        elif action == "turn_right":
            theta -= TURN_DELTA
            node.set_rotation([0.0, 0.0, 1.0, theta])


# --- Mock del módulo compilado `managers` -----------------------------------

class FakeGaitManager:
    """Mock de RobotisOp2GaitManager."""

    def __init__(self, robot, config_path):
        self.robot = robot
        self.config_path = config_path
        self._started = False
        self.x_amplitude = 0.0
        self.a_amplitude = 0.0

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def setXAmplitude(self, value):
        self.x_amplitude = value

    def setAAmplitude(self, value):
        self.a_amplitude = value

    def setBalanceEnable(self, value):
        pass

    def step(self, time_step):
        pass


class FakeMotionManager:
    """Mock de RobotisOp2MotionManager."""

    def __init__(self, robot, config_path):
        self.robot = robot
        self.config_path = config_path
        self.last_page = None

    def playPage(self, page):
        self.last_page = page

    def step(self, time_step):
        pass


def _build_managers_module() -> types.ModuleType:
    module = types.ModuleType("managers")
    module.RobotisOp2GaitManager = FakeGaitManager
    module.RobotisOp2MotionManager = FakeMotionManager
    return module


def _build_controller_module(supervisor_factory) -> types.ModuleType:
    module = types.ModuleType("controller")
    module.Supervisor = supervisor_factory
    module.Robot = supervisor_factory
    module.Node = FakeNode
    module.Field = FakeField
    module.Accelerometer = FakeSensor
    module.Gyro = FakeSensor
    module.Motor = FakeMotor
    return module


class FakeWebots:
    """Controlador del mock: fabrica supervisors e instala los módulos.

    `install()` registra los módulos `controller` y `managers` en sys.modules
    para que los imports de producción resuelvan al mock. `make_supervisor`
    crea un Supervisor para un robot concreto por nombre.
    """

    def __init__(self):
        FakeSupervisor._world = {
            name: FakeNode(name, translation) for name, translation in _ROBOT_STARTS.items()
        }
        for name, translation in _BARRIERS.items():
            FakeSupervisor._world[name] = FakeNode(name, translation)
        FakeSupervisor._root = FakeRoot(list(FakeSupervisor._world.values()))
        self._installed = []

    def make_supervisor(self, self_name: str = "DQN", time_step: int = 32) -> FakeSupervisor:
        return FakeSupervisor(self_name=self_name, time_step=time_step)

    def node(self, name: str) -> FakeNode:
        return FakeSupervisor._world[name]

    def install(self) -> None:
        """Instala `controller` y `managers` mock en sys.modules."""
        self._saved = {name: sys.modules.get(name) for name in ("controller", "managers")}
        sys.modules["controller"] = _build_controller_module(self.make_supervisor)
        sys.modules["managers"] = _build_managers_module()

    def uninstall(self) -> None:
        for name, mod in getattr(self, "_saved", {}).items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture
def fake_webots():
    """Fixture: mundo mock fresco con los módulos controller/managers instalados."""
    fw = FakeWebots()
    fw.install()
    try:
        yield fw
    finally:
        fw.uninstall()
