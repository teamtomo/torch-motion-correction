from pathlib import Path
from typing import Sequence
from torch_fourier_filter.bandpass import bandpass_filter

import torch


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


def fftfreq_to_spatial_frequency(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per pixel to cycles per unit distance."""
    # cycles/px * px/distance = cycles/distance
    return torch.as_tensor(frequencies, dtype=torch.float32) * (1 / spacing)


def spatial_frequency_to_fftfreq(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per unit distance to cycles per pixel."""
    # cycles/distance * distance/px = cycles/px
    return torch.as_tensor(frequencies, dtype=torch.float32) * spacing

def normalize_image(image: torch.Tensor):
    # grab image dimensions
    t, h, w = image.shape

    # calculate limits of central box
    hl, hu = int(0.25 * h), int(0.75 * h)
    wl, wu = int(0.25 * w), int(0.75 * w)

    # calculate mean and std of central 50%
    center = image[:, hl:hu, wl:wu]
    std, mean = torch.std_mean(center, dim=(-3, -2, -1))

    # normalize and return
    image = (image - mean) / std
    return image

def prepare_bandpass_filter(
    frequency_range: tuple[float, float],  # angstroms
    patch_shape: tuple[int, int],
    pixel_spacing: float,  # angstroms
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare bandpass filter for cross-correlation (fixed, no refinement)."""
    ph, pw = patch_shape
    
    # Use the higher resolution cutoff (smaller angstrom value)
    cuton, cutoff = frequency_range
    low_fftfreq = spatial_frequency_to_fftfreq(1 / cuton, spacing=pixel_spacing)
    high_fftfreq = spatial_frequency_to_fftfreq(1 / cutoff, spacing=pixel_spacing)
    
    # Prepare bandpass
    bandpass = bandpass_filter(
        low=low_fftfreq,
        high=high_fftfreq,
        falloff=0,
        image_shape=(ph, pw),
        rfft=True,
        fftshift=False,
        device=device
    )
    
    return bandpass

