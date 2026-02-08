"""State and command enums used by the drone controller."""
from __future__ import annotations
from enum import Enum, auto

class CurrentCommand(Enum):
    """High-level commands derived from gesture recognition."""
    TAKE_OFF = auto()
    IDLE = auto()
    FLY_LEFT = auto()
    FLY_RIGHT = auto()
    FLY_UP = auto()
    FLY_DOWN = auto()
    FLY_BACK = auto()
    FLY_FORWARD = auto()
    HALT_MID_AIR = auto()
    LANDING = auto()
    FLIP = auto()
    ROTATE = auto()
    XY = auto()
