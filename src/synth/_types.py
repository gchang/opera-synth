from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import TYPE_CHECKING, NamedTuple, ParamSpec, TypeVar

# Some classes are declared as generic in stubs, but not at runtime.
# In Python 3.9 and earlier, os.PathLike is not subscriptable, results in runtime error
if TYPE_CHECKING:
    from builtins import ellipsis

    Index = ellipsis | slice | int
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike

P = ParamSpec("P")
T = TypeVar("T")


class Bbox(NamedTuple):
    """Bounding box named tuple, defining extent in cartesian coordinates.

    Usage:

        Bbox(left, bottom, right, top)

    Attributes
    ----------
    left : float
        Left coordinate (xmin)
    bottom : float
        Bottom coordinate (ymin)
    right : float
        Right coordinate (xmax)
    top : float
        Top coordinate (ymax)

    """

    left: float
    bottom: float
    right: float
    top: float


PathOrStr = str | PathLikeStr
# TypeVar added for generic functions which should return the same type as the input
PathLikeT = TypeVar("PathLikeT", str, PathLikeStr)


class Season(Enum):
    """Seasonal periods for the year."""

    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"


class Variable(Enum):
    """Variables for the global coherence raster."""

    AMP = "amp"  # note: capitalized in the dataset
    TAU = "tau"
    RHO = "rho"
    RMSE = "rmse"


class RhoOption(Enum):
    """Options to transform `rho` for noiser/less noisy data."""

    SHRUNK = "shrunk"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
