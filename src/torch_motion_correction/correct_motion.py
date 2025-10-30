import torch
import torch.nn.functional as F
import einops
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d
from torch_fourier_shift import fourier_shift_dft_2d

from torch_motion_correction.deformation_field_utils import (
    evaluate_deformation_field, evaluate_deformation_field_at_t,
)

from torch_image_interpolation.grid_sample_utils import array_to_grid_sample


def correct_motion(
    image: torch.Tensor,  # (t, h, w)
    deformation_grid: torch.Tensor,  # (yx, t, h, w)
    pixel_spacing: float,
    grad: bool = False,
    grid_type: str = "catmull_rom",
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        deformation_grid = deformation_grid.to(device)

    t, _, _ = image.shape
    _, gt, gh, gw = deformation_grid.shape
    normalized_t = torch.linspace(0, 1, steps=t, device=image.device)

    # Use conditional gradient context to save memory
    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        # correct motion in each frame
        corrected_frames = [
            _correct_frame(
                frame=frame,
                frame_deformation_grid=evaluate_deformation_field_at_t(
                    deformation_field=deformation_grid,
                    t=frame_t,
                    grid_shape=(10 * gh, 10 * gw),
                    grid_type=grid_type,
                ),
                pixel_spacing=pixel_spacing,
            )
            for frame, frame_t
            in zip(image, normalized_t)
        ]
    corrected_frames = torch.stack(corrected_frames, dim=0).detach()
    return corrected_frames  # (t, h, w)


def _correct_frame(
    frame: torch.Tensor,
    pixel_spacing: float,
    frame_deformation_grid: torch.Tensor,  # (yx, h, w)
) -> torch.Tensor:
    # grab frame and deformation grid dimensions
    h, w = frame.shape
    _, gh, gw = frame_deformation_grid.shape

    # prepare a grid of pixel positions
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=frame.device,
    )  # (h, w, 2) yx coords

    # interpolate oversampled per frame deformation grid at each pixel position
    image_dim_lengths = torch.as_tensor([h - 1, w - 1], device=frame.device, dtype=torch.float32)
    deformation_grid_dim_lengths = torch.as_tensor([gh - 1, gw - 1], device=frame.device, dtype=torch.float32)
    normalized_pixel_grid = pixel_grid / image_dim_lengths
    deformation_grid_interpolants = normalized_pixel_grid * deformation_grid_dim_lengths
    deformation_grid_interpolants = array_to_grid_sample(deformation_grid_interpolants, array_shape=(gh, gw)) # (gh, gw, xy)
    shifts_angstroms = F.grid_sample(
        input=einops.rearrange(frame_deformation_grid, 'yx h w -> 1 yx h w'),
        grid=einops.rearrange(deformation_grid_interpolants, 'h w xy -> 1 h w xy'),
        mode='bicubic',
        padding_mode='reflection',
        align_corners=True,
    )  # (b, yx, h, w)

    pixel_shifts = shifts_angstroms / pixel_spacing
    # find pixel positions to sample image data at, accounting for deformations
    pixel_shifts = einops.rearrange(pixel_shifts, '1 yx h w -> h w yx')

    # todo: make sure semantics around deformation field interpolants (i.e. spatiotemporally resolved shifts) are crystal clear
    deformed_pixel_coords = pixel_grid + pixel_shifts

    # sample original image data
    corrected_frame = sample_image_2d(
        image=frame,
        coordinates=deformed_pixel_coords,
        interpolation='bicubic',
    )

    return corrected_frame

def correct_motion_slow(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    grad: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        deformation_grid = deformation_grid.to(device)
        
    t, _, _ = image.shape
    normalized_t = torch.linspace(0, 1, steps=t, device=image.device)
    
        
    # Use conditional gradient context to save memory
    gradient_context = torch.enable_grad() if grad else torch.no_grad()
    
    with gradient_context:
        # correct motion in each frame
        corrected_frames = [
            _correct_frame_slow(
                frame=frame,
                deformation_grid=deformation_grid,
                t=frame_t,
            )
            for frame, frame_t
            in zip(image, normalized_t)
        ]
    corrected_frames = torch.stack(corrected_frames, dim=0).detach()
    return corrected_frames # (t, h, w)


def _correct_frame_slow(
    frame: torch.Tensor,
    deformation_grid: torch.Tensor,
    t: float  # [0, 1]
) -> torch.Tensor:

    # grab frame dimensions
    h, w = frame.shape

    # prepare a grid of pixel positions
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=frame.device,
    )  # (h, w, 2) yx coords

    dim_lengths = torch.as_tensor([h - 1, w - 1], device=frame.device, dtype=torch.float32)
    normalized_pixel_grid = pixel_grid / dim_lengths

    # add normalized time coordinate to every pixel coordinate
    # (h, w, 2) -> (h, w, 3)
    # yx -> tyx
    tyx = F.pad(normalized_pixel_grid, pad=(1, 0), value=t)

    # evaluate interpolated shifts at every pixel
    shifts_px = evaluate_deformation_field(
        deformation_field=deformation_grid,
        tyx=tyx,
    )

    # find pixel positions to sample image data at, accounting for deformations
    deformed_pixel_coords = pixel_grid + shifts_px

    # sample original image data
    corrected_frame = sample_image_2d(
        image=frame,
        coordinates=deformed_pixel_coords,
        interpolation='bicubic',
    )

    return corrected_frame


def correct_motion_batched(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    batch_size: int = None,
    grad: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Correct motion for all frames using batched processing to speed up evaluate_deformation_grid.
    
    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_grid: torch.Tensor
        Deformation grid for motion correction
    batch_size: int, optional
        Number of frames to process at once. If None, processes all frames at once.
    grad: bool
        Whether to enable gradients
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) corrected images
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        deformation_grid = deformation_grid.to(device)

    t, h, w = image.shape

    # Use conditional gradient context to save memory
    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        if batch_size is None or batch_size >= t:
            # Process all frames at once
            corrected_frames = _correct_frames(
                frames=image,
                deformation_grid=deformation_grid,
            )
        else:
            # Process in batches
            corrected_frames = []
            for start_idx in range(0, t, batch_size):
                end_idx = min(start_idx + batch_size, t)
                batch_frames = image[start_idx:end_idx]
                batch_corrected = _correct_frames(
                    frames=batch_frames,
                    deformation_grid=deformation_grid,
                    frame_offset=start_idx,
                )
                corrected_frames.append(batch_corrected)
            corrected_frames = torch.cat(corrected_frames, dim=0)

    return corrected_frames.detach()


def _correct_frames(
    frames: torch.Tensor,  # (t, h, w) or (batch_t, h, w)
    deformation_grid: torch.Tensor,
    frame_offset: int = 0,
) -> torch.Tensor:
    """
    Correct motion for a batch of frames at once.
    
    Parameters
    ----------
    frames: torch.Tensor
        (t, h, w) or (batch_t, h, w) array of frames to correct
    deformation_grid: torch.Tensor
        Deformation grid for motion correction
    frame_offset: int
        Offset for frame indices (for batching)
        
    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) or (batch_t, h, w) corrected frames
    """

    batch_t, h, w = frames.shape

    # Prepare coordinate grid once for all frames
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=frames.device,
    )  # (h, w, 2) yx coords

    # Normalize pixel grid once
    dim_lengths = torch.as_tensor([h - 1, w - 1], device=frames.device, dtype=torch.float32)
    normalized_pixel_grid = pixel_grid / dim_lengths

    # Create normalized time coordinates for all frames
    normalized_t = torch.linspace(0, 1, steps=batch_t, device=frames.device)
    if frame_offset > 0:
        # Adjust time coordinates for batch offset
        total_frames = deformation_grid.shape[1]  # Assuming deformation_grid is (2, t, h, w)
        normalized_t = torch.linspace(frame_offset / (total_frames - 1),
                                      (frame_offset + batch_t - 1) / (total_frames - 1),
                                      steps=batch_t, device=frames.device
                                      )

    # Add time coordinate to pixel grid for all frames at once
    # (h, w, 2) -> (batch_t, h, w, 3)
    tyx = normalized_pixel_grid.unsqueeze(0).expand(batch_t, -1, -1, -1)  # (batch_t, h, w, 2)
    t_coords = normalized_t.view(-1, 1, 1, 1).expand(-1, h, w, 1)  # (batch_t, h, w, 1)
    tyx = torch.cat([t_coords, tyx], dim=-1)  # (batch_t, h, w, 3)

    # Evaluate deformation grid for all frames at once
    # Reshape to (batch_t * h * w, 3) for batch evaluation
    tyx_flat = tyx.view(-1, 3)  # (batch_t * h * w, 3)
    shifts_px_flat = evaluate_deformation_field(
        deformation_field=deformation_grid,
        tyx=tyx_flat,
    )  # (batch_t * h * w, 2)

    # Reshape back to (batch_t, h, w, 2)
    shifts_px = shifts_px_flat.view(batch_t, h, w, 2)

    # Find deformed pixel coordinates for all frames
    pixel_grid_expanded = pixel_grid.unsqueeze(0).expand(batch_t, -1, -1, -1)  # (batch_t, h, w, 2)
    deformed_pixel_coords = pixel_grid_expanded + shifts_px  # (batch_t, h, w, 2)

    # Sample image data for all frames
    corrected_frames = []
    for i in range(batch_t):
        corrected_frame = sample_image_2d(
            image=frames[i],
            coordinates=deformed_pixel_coords[i],
            interpolation='bicubic',
        )
        corrected_frames.append(corrected_frame)

    corrected_frames = torch.stack(corrected_frames, dim=0)

    return corrected_frames


def correct_motion_fast(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Fast motion correction for single patch deformation fields using FFT.
    
    This function checks that the deformation field represents a single patch
    (final two dimensions are both 1) and uses FFT-based correction with
    fourier_shift_dft_2d for efficiency.
    
    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_grid: torch.Tensor
        Deformation field (2, t, 1, 1) for single patch correction
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) corrected images
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        deformation_grid = deformation_grid.to(device)

    # Check that deformation field has single patch dimensions
    if deformation_grid.shape[-2:] != (1, 1):
        raise ValueError(f"Expected single patch deformation field with shape (2, t, 1, 1), "
                         f"but got shape {deformation_grid.shape}. "
                         f"Final two dimensions must be (1, 1) for single patch correction."
                         )

    t, h, w = image.shape

    # Extract shifts from deformation field (2, t, 1, 1) -> (t, 2)
    shifts = einops.rearrange(deformation_grid, 'c t 1 1 -> t c')
    shifts *= -1  # flip for phase shift

    print(f"Single patch correction: applying shifts to {t} frames")
    print(f"Shift range: y=[{shifts[:, 0].min():.2f}, {shifts[:, 0].max():.2f}], "
          f"x=[{shifts[:, 1].min():.2f}, {shifts[:, 1].max():.2f}] pixels"
          )

    # Convert image to frequency domain for efficient shifting

    image_fft = torch.fft.rfftn(image, dim=(-2, -1))  # (t, h, w_freq)

    # Apply shifts using fourier_shift_dft_2d

    shifted_fft = fourier_shift_dft_2d(
        dft=image_fft,
        image_shape=(h, w),
        shifts=shifts,  # (t, 2) shifts
        rfft=True,
        fftshifted=False,
    )

    corrected_frames = torch.fft.irfftn(shifted_fft, s=(h, w))

    return corrected_frames