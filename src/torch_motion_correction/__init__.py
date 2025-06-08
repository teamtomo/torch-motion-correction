"""Motion estimation and correction in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-motion-correction")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_motion_correction.estimate_motion import estimate_motion
from torch_motion_correction.correct_motion import correct_motion
from torch_motion_correction.evaluate_deformation_grid import evaluate_deformation_grid

__all__ = [
    "estimate_motion",
    "correct_motion",
    "evaluate_deformation_grid",
]
