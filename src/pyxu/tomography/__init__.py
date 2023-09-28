from pyxu.info.plugin import _load_entry_points
from pyxu.tomography.kernel import *
from pyxu.tomography.xray import *

_load_entry_points(globals(), group="pyxu.tomography")
