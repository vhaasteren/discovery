"""Discovery"""
from __future__ import annotations

import jax.config
jax.config.update("jax_enable_x64", True)

from .const import *
from .matrix import *
from .prior import *
from .signals import *
from .likelihood import *
from .solar import *

__version__ = "0.1"
