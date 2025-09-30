import einops
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicCatmullRomGrid3d


def evaluate_deformation_field(
    deformation_field: torch.Tensor,  # (yx, nt, nh, nw)
    tyx: torch.Tensor,  # (..., 3)
) -> torch.Tensor:  # (
    """Evaluate shifts from deformation field data."""
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field).to(deformation_field.device)
    predicted_shifts = deformation_field(tyx)
    return predicted_shifts


def evaluate_deformation_field_at_t(
    deformation_field: torch.Tensor,  # (yx, nt, nh, nw)
    t: float,  # [0, 1]
    grid_shape: tuple[int, int],  # (h, w)
) -> torch.Tensor:
    """Evaluate a grid of shifts at a specific timepoint from deformation field data.

     output: (tyx, h, w)
     """
    h, w = grid_shape
    y = torch.linspace(0, 1, steps=h, device=deformation_field.device)
    x = torch.linspace(0, 1, steps=w, device=deformation_field.device)

    # Create meshgrid and flatten to get (h*w, 2) array of yx coordinates
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")

    # Add t coordinate using F.pad to create tyx triplets
    tyx_grid = F.pad(yx_grid, (1, 0), value=t)  # (h*w, 3)

    # Evaluate deformation grid and return as (tyx, h, w)
    shifts = evaluate_deformation_field(deformation_field, tyx=tyx_grid)  # (h*w, c)
    shifts = einops.rearrange(shifts, "(h w) tyx -> tyx h w", h=h, w=w)
    return shifts

def resample_deformation_field(
    deformation_field: torch.Tensor,
    target_resolution: tuple[int, int, int],
) -> torch.Tensor:
    nt, nh, nw = target_resolution

    # setup grid of points to evaluate over existing deformation grid
    t = torch.linspace(0, 1, steps=nt)
    y = torch.linspace(0, 1, steps=nh)
    x = torch.linspace(0, 1, steps=nw)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing='ij')
    tyx = einops.rearrange([tt, yy, xx], "tyx nt nh nw -> nt nh nw tyx")

    # evaluate existing field at new field grid points
    new_deformation_field = evaluate_deformation_field(deformation_field, tyx=tyx)

    return new_deformation_field