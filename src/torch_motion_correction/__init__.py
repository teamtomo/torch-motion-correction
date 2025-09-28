"""Motion estimation and correction in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-motion-correction")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_motion_correction.estimate_motion import estimate_motion, estimate_motion_lazy
from torch_motion_correction.correct_motion import (
    correct_motion, 
    correct_motion_batched, 
    correct_motion_fast,
)
from torch_motion_correction.evaluate_deformation_grid import (
    evaluate_deformation_grid, 
)
from torch_motion_correction.estimate_motion_cross_correlation import (
    estimate_motion_cross_correlation_whole_image,
    estimate_motion_cross_correlation_patches,
)

from torch_motion_correction.data_io import (
    write_deformation_field_to_csv,
    read_deformation_field_from_csv,
)


__all__ = [
    "estimate_motion",
    "estimate_motion_lazy",
    "correct_motion",
    "correct_motion_batched",
    "correct_motion_fast",
    "evaluate_deformation_grid",
    "estimate_motion_cross_correlation_whole_image",
    "estimate_motion_cross_correlation_patches",
    "write_deformation_field_to_csv",
    "read_deformation_field_from_csv",
]
