import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d


def evaluate_deformation_grid(
    deformation_grid: torch.Tensor,  # (nt, nh, nw)
    tyx: torch.Tensor,  # (..., 3)
) -> torch.Tensor:
    """Evaluate shifts from deformation field data."""
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_grid)
    predicted_shifts = deformation_field(tyx)
    return predicted_shifts
