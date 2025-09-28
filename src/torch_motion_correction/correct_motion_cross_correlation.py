import torch
import torch.nn.functional as F
import numpy as np
import einops
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d


def correct_motion_whole_image(
    image: torch.Tensor,  # (t, h, w)
    shifts: torch.Tensor,  # (t, 2) shifts in pixels (y, x)
    fill_mode: str = 'noise',  # 'noise', 'zeros', 'edge'
    device: torch.device = None,
) -> torch.Tensor:
    """
    Apply motion correction using whole image shifts.
    
    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to correct
    shifts: torch.Tensor
        (t, 2) array of shifts for each frame in pixels (y, x)
    fill_mode: str
        How to handle edges: 'noise' (random with same stats), 'zeros', 'edge'
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    corrected_image: torch.Tensor
        (t, h, w) motion corrected images
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        shifts = shifts.to(device)
    
    t, h, w = image.shape
    
    # Create coordinate grid
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=device,
    )  # (h, w, 2) yx coords
    
    corrected_frames = []
    
    for frame_idx in range(t):
        frame = image[frame_idx]
        frame_shifts = shifts[frame_idx]  # (2,) - (y, x)
        
        # Calculate sampling coordinates
        sampling_coords = pixel_grid + frame_shifts  # (h, w, 2)
        
        if fill_mode == 'noise':
            # Create noise with same statistics as the frame
            corrected_frame = _sample_with_noise_fill(frame, sampling_coords)
        elif fill_mode == 'zeros':
            corrected_frame = sample_image_2d(
                image=frame,
                coordinates=sampling_coords,
                interpolation='bicubic',
            )
        elif fill_mode == 'edge':
            corrected_frame = sample_image_2d(
                image=frame,
                coordinates=sampling_coords,
                interpolation='bicubic',
            )
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")
            
        corrected_frames.append(corrected_frame)
    
    return torch.stack(corrected_frames, dim=0)


def _sample_with_noise_fill(
    image: torch.Tensor,  # (h, w)
    coordinates: torch.Tensor,  # (h, w, 2)
) -> torch.Tensor:
    """
    Sample image with noise filling for out-of-bounds regions.
    
    The noise has the same mean and standard deviation as the input image.
    """
    h, w = image.shape
    device = image.device
    
    # Calculate image statistics
    img_mean = image.mean()
    img_std = image.std()
    
    # Create noise image with same statistics
    noise_image = torch.randn_like(image) * img_std + img_mean
    
    # Find out-of-bounds coordinates
    y_coords = coordinates[:, :, 0]  # (h, w)
    x_coords = coordinates[:, :, 1]  # (h, w)
    
    oob_mask = (
        (y_coords < 0) | (y_coords >= h) |
        (x_coords < 0) | (x_coords >= w)
    )
    
    # Sample the original image
    sampled = sample_image_2d(
        image=image,
        coordinates=coordinates,
        interpolation='bicubic',
    )
    
    # Replace out-of-bounds regions with noise
    sampled[oob_mask] = noise_image[oob_mask]
    
    return sampled


def create_deformation_field_from_whole_image_shifts(
    shifts: torch.Tensor,  # (t, 2) shifts in pixels
    image_shape: tuple[int, int, int],  # (t, h, w)
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert whole image shifts to a deformation field for compatibility.
    
    Parameters
    ----------
    shifts: torch.Tensor
        (t, 2) array of shifts for each frame in pixels (y, x)
    image_shape: tuple[int, int, int]
        Shape of the image (t, h, w)
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    deformation_field: torch.Tensor
        (2, t, 1, 1) deformation field with constant shifts per frame
    """
    if device is None:
        device = shifts.device
    else:
        shifts = shifts.to(device)
    
    t, h, w = image_shape
    
    # Create deformation field with constant shifts per frame
    # Shape: (2, t, 1, 1) - 2 channels (y, x), t time points, 1x1 spatial
    # Expand the shifts to (2, t, 32, 32) so that each frame has a constant shift over a 32x32 grid
    # shifts: (t, 2) -> (2, t, 1, 1) -> (2, t, 32, 32)
    # Use einops to reshape and repeat the shifts tensor
    # shifts: (t, 2) -> (2, t, 1, 1)
    deformation_field = einops.rearrange(shifts, 't c -> c t 1 1')
    # (2, t, 1, 1) -> (2, t, 32, 32)
    deformation_field = einops.repeat(deformation_field, 'c t 1 1 -> c t h w', h=32, w=32)
    
    return deformation_field

