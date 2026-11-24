"""Tests del controlador de prueba locomotion_test (sin RL, sin Webots).

Verifica la resolución del comando con prioridad controllerArgs (argv[1]) ->
LOCO_CMD -> 'stand'. El módulo importa `config` y `darwin_robot` (en el
pythonpath de pytest); su `main()` sólo toca `controller` dentro de la función,
por lo que importar el módulo y llamar a `_read_command` es seguro sin Webots.
"""

import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODULE_PATH = os.path.join(_ROOT, "controllers", "locomotion_test", "locomotion_test.py")


def _load_locomotion_test():
    spec = importlib.util.spec_from_file_location("locomotion_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loco(fake_webots):
    # fake_webots instala el módulo `controller` mock por si el import lo usa.
    return _load_locomotion_test()


def test_command_from_controller_args_first(loco, monkeypatch):
    monkeypatch.setenv("LOCO_CMD", "left")  # debe perder frente a argv[1]
    assert loco._read_command(["locomotion_test", "forward"]) == "forward"


def test_command_from_env_when_no_args(loco, monkeypatch):
    monkeypatch.setenv("LOCO_CMD", "right")
    assert loco._read_command(["locomotion_test"]) == "right"


def test_command_defaults_to_stand(loco, monkeypatch):
    monkeypatch.delenv("LOCO_CMD", raising=False)
    assert loco._read_command(["locomotion_test"]) == "stand"


def test_command_invalid_falls_back_to_stand(loco, monkeypatch):
    monkeypatch.delenv("LOCO_CMD", raising=False)
    assert loco._read_command(["locomotion_test", "jump"]) == "stand"


def test_all_valid_commands_map_to_methods(loco):
    assert set(loco._COMMANDS) == {"stand", "forward", "left", "right"}
    assert loco._COMMANDS["forward"] == "walk_forward"
    assert loco._COMMANDS["left"] == "turn_left"
    assert loco._COMMANDS["right"] == "turn_right"
    assert loco._COMMANDS["stand"] == "stand"
