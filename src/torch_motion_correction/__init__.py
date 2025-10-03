"""Motion estimation and correction in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-motion-correction")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_motion_correction.estimate_local_motion import estimate_local_motion
from torch_motion_correction.correct_motion import (
    correct_motion, 
    correct_motion_batched, 
    correct_motion_fast,
    correct_motion_slow,
)
from torch_motion_correction.deformation_field_utils import (
    evaluate_deformation_field,
)
from torch_motion_correction.estimate_global_motion import (
    estimate_global_motion,
    estimate_motion_cross_correlation_patches,
)

from torch_motion_correction.data_io import (
    write_deformation_field_to_csv,
    read_deformation_field_from_csv,
)


__all__ = [
    "estimate_local_motion",
    "correct_motion",
    "correct_motion_batched",
    "correct_motion_fast",
    "correct_motion_slow",
    "evaluate_deformation_field",
    "estimate_global_motion",
    "estimate_motion_cross_correlation_patches",
    "write_deformation_field_to_csv",
    "read_deformation_field_from_csv",
]
