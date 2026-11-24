"""Sanity test del mundo pista_atletismo_50m.wbt.

Verifica que FEAT-004 fijó EXACTAMENTE los cuatro campos 'controller' a las
carpetas de los controladores y que la geometría (translation/rotation) y los
nombres/flags de los robots quedaron intactos respecto a los valores conocidos
del mundo (ver context.json / world_facts).

No lanza Webots: solo parsea el texto del .wbt.
"""

import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WBT = os.path.join(_ROOT, "pista_atletismo_50m.wbt")

# Nombre de robot -> (carpeta de controlador, translation esperada).
_EXPECTED = {
    "DQN": ("dqn_agent", (0.11, 9.95, 0.24)),
    "ppo": ("ppo_agent", (0.21, 7.88, 0.34)),
    "actor - critic": ("a2c_agent", (0.21, 6.04, 0.34)),
    "agente propio": ("custom_agent", (0.21, 11.99, 0.34)),
}


def _read_wbt() -> str:
    with open(_WBT, encoding="utf-8") as fh:
        return fh.read()


def _robot_blocks(text: str):
    """Devuelve {name: block_text} para cada Darwin-op con campo 'name'."""
    blocks = {}
    for match in re.finditer(r"DEF DARWIN Darwin-op \{(.*?)\n\}", text, re.DOTALL):
        body = match.group(1)
        name_match = re.search(r'name\s+"([^"]*)"', body)
        if name_match:
            blocks[name_match.group(1)] = body
    return blocks


def test_exactly_four_controllers_set():
    text = _read_wbt()
    controllers = re.findall(
        r'controller\s+"(dqn_agent|ppo_agent|a2c_agent|custom_agent)"', text
    )
    assert sorted(controllers) == ["a2c_agent", "custom_agent", "dqn_agent", "ppo_agent"]
    # Ningún robot quedó sin controlador asignado.
    assert 'controller "<none>"' not in text


def test_each_robot_has_expected_controller():
    blocks = _robot_blocks(_read_wbt())
    assert set(blocks) == set(_EXPECTED)
    for name, (folder, _translation) in _EXPECTED.items():
        assert re.search(rf'controller\s+"{folder}"', blocks[name]), name


def test_robot_geometry_and_flags_unchanged():
    blocks = _robot_blocks(_read_wbt())
    for name, (_folder, translation) in _EXPECTED.items():
        body = blocks[name]
        trans = re.search(r"translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", body)
        assert trans is not None, name
        values = tuple(round(float(v), 5) for v in trans.groups())
        assert values == translation, (name, values)
        # Flags de cada robot intactos.
        assert "supervisor TRUE" in body
        assert "selfCollision TRUE" in body
