import einops
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicCatmullRomGrid3d


def evaluate_deformation_grid(
    deformation_grid: torch.Tensor,  # (c, nt, nh, nw)
    tyx: torch.Tensor,  # (..., 3)
) -> torch.Tensor:  # (
    """Evaluate shifts from deformation field data."""
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_grid).to(deformation_grid.device)
    predicted_shifts = deformation_field(tyx)
    return predicted_shifts


def evaluate_deformation_grid_at_t(
    deformation_grid: torch.Tensor,  # (c, nt, nh, nw)
    t: float,  # [0, 1]
    grid_shape: tuple[int, int],  # (h, w)
) -> torch.Tensor:
    """Evaluate a grid of shifts at a specific timepoint from deformation field data.

     output: (tyx, h, w)
     """
    h, w = grid_shape
    y = torch.linspace(0, 1, steps=h, device=deformation_grid.device)
    x = torch.linspace(0, 1, steps=w, device=deformation_grid.device)

    # Create meshgrid and flatten to get (h*w, 2) array of yx coordinates
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")

    # Add t coordinate using F.pad to create tyx triplets
    tyx_grid = F.pad(yx_grid, (1, 0), value=t)  # (h*w, 3)

    # Evaluate deformation grid and return as (tyx, h, w)
    shifts = evaluate_deformation_grid(deformation_grid, tyx_grid)  # (h*w, c)
    shifts = einops.rearrange(shifts, "(h w) tyx -> tyx h w", h=h, w=w)
    return shifts
