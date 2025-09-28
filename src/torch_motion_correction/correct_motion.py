import torch
import torch.nn.functional as F
import einops
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d
from torch_fourier_shift import fourier_shift_dft_2d

from torch_motion_correction.evaluate_deformation_grid import evaluate_deformation_grid_lazy, evaluate_deformation_grid

import time

def correct_motion(
    image: torch.Tensor,
    pixel_spacing: float,
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
            _correct_frame(
                frame=frame,
                pixel_spacing=pixel_spacing,
                deformation_grid=deformation_grid,
                t=frame_t,
            )
            for frame, frame_t
            in zip(image, normalized_t)
        ]
    corrected_frames = torch.stack(corrected_frames, dim=0).detach()
    return corrected_frames # (t, h, w)


def _correct_frame(
    frame: torch.Tensor,
    pixel_spacing: float,
    deformation_grid: torch.Tensor,
    t: float  # [0, 1]
) -> torch.Tensor:

    if frame.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()

    # grab frame dimensions
    h, w = frame.shape

    # prepare a grid of pixel positions
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=frame.device,
    )  # (h, w, 2) yx coords
    if frame.is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()

    dim_lengths = torch.as_tensor([h - 1, w - 1], device=frame.device, dtype=torch.float32)
    normalized_pixel_grid = pixel_grid / dim_lengths
    if frame.is_cuda:
        torch.cuda.synchronize()
    t2 = time.time()

    # add normalized time coordinate to every pixel coordinate
    # (h, w, 2) -> (h, w, 3)
    # yx -> tyx
    tyx = F.pad(normalized_pixel_grid, pad=(1, 0), value=t)
    if frame.is_cuda:
        torch.cuda.synchronize()
    t3 = time.time()

    # evaluate interpolated shifts at every pixel
    #evaluator = evaluate_deformation_grid_lazy(
    #    deformation_grid=deformation_grid,
    #    coordinates=tyx,
    #)
    # evaluate interpolated shifts at every pixel
    shifts_angstroms = evaluate_deformation_grid(
        deformation_grid=deformation_grid,
        tyx=tyx,
    )
    # Extract the actual tensor values from the lazy evaluator
    #shifts_angstroms = evaluator.evaluate_all()
    shifts_px = shifts_angstroms
    if frame.is_cuda:
        torch.cuda.synchronize()
    t4 = time.time()
    # find pixel positions to sample image data at, accounting for deformations
    deformed_pixel_coords = pixel_grid + shifts_px
    if frame.is_cuda:
        torch.cuda.synchronize()
    t5 = time.time()

    # sample original image data
    corrected_frame = sample_image_2d(
        image=frame,
        coordinates=deformed_pixel_coords,
        interpolation='bicubic',
    )
    if frame.is_cuda:
        torch.cuda.synchronize()
    t6 = time.time()

    return corrected_frame


def correct_motion_batched(
    image: torch.Tensor,
    pixel_spacing: float,
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
    pixel_spacing: float
        Pixel spacing in angstroms
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
                pixel_spacing=pixel_spacing,
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
                    pixel_spacing=pixel_spacing,
                    deformation_grid=deformation_grid,
                    frame_offset=start_idx,
                )
                corrected_frames.append(batch_corrected)
            corrected_frames = torch.cat(corrected_frames, dim=0)
    
    return corrected_frames.detach()


def _correct_frames(
    frames: torch.Tensor,  # (t, h, w) or (batch_t, h, w)
    pixel_spacing: float,
    deformation_grid: torch.Tensor,
    frame_offset: int = 0,
) -> torch.Tensor:
    """
    Correct motion for a batch of frames at once.
    
    Parameters
    ----------
    frames: torch.Tensor
        (t, h, w) or (batch_t, h, w) array of frames to correct
    pixel_spacing: float
        Pixel spacing in angstroms
    deformation_grid: torch.Tensor
        Deformation grid for motion correction
    frame_offset: int
        Offset for frame indices (for batching)
        
    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) or (batch_t, h, w) corrected frames
    """
    if frames.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    
    batch_t, h, w = frames.shape
    
    # Prepare coordinate grid once for all frames
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=frames.device,
    )  # (h, w, 2) yx coords
    
    if frames.is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    
    # Normalize pixel grid once
    dim_lengths = torch.as_tensor([h - 1, w - 1], device=frames.device, dtype=torch.float32)
    normalized_pixel_grid = pixel_grid / dim_lengths
    
    if frames.is_cuda:
        torch.cuda.synchronize()
    t2 = time.time()
    
    # Create normalized time coordinates for all frames
    normalized_t = torch.linspace(0, 1, steps=batch_t, device=frames.device)
    if frame_offset > 0:
        # Adjust time coordinates for batch offset
        total_frames = deformation_grid.shape[1]  # Assuming deformation_grid is (2, t, h, w)
        normalized_t = torch.linspace(frame_offset / (total_frames - 1), 
                                   (frame_offset + batch_t - 1) / (total_frames - 1), 
                                   steps=batch_t, device=frames.device)
    
    # Add time coordinate to pixel grid for all frames at once
    # (h, w, 2) -> (batch_t, h, w, 3)
    tyx = normalized_pixel_grid.unsqueeze(0).expand(batch_t, -1, -1, -1)  # (batch_t, h, w, 2)
    t_coords = normalized_t.view(-1, 1, 1, 1).expand(-1, h, w, 1)  # (batch_t, h, w, 1)
    tyx = torch.cat([t_coords, tyx], dim=-1)  # (batch_t, h, w, 3)
    
    if frames.is_cuda:
        torch.cuda.synchronize()
    t3 = time.time()
    
    # Evaluate deformation grid for all frames at once
    # Reshape to (batch_t * h * w, 3) for batch evaluation
    tyx_flat = tyx.view(-1, 3)  # (batch_t * h * w, 3)
    shifts_angstroms_flat = evaluate_deformation_grid(
        deformation_grid=deformation_grid,
        tyx=tyx_flat,
    )  # (batch_t * h * w, 2)
    
    # Reshape back to (batch_t, h, w, 2)
    shifts_angstroms = shifts_angstroms_flat.view(batch_t, h, w, 2)
    shifts_px = shifts_angstroms
    
    if frames.is_cuda:
        torch.cuda.synchronize()
    t4 = time.time()
    
    # Find deformed pixel coordinates for all frames
    pixel_grid_expanded = pixel_grid.unsqueeze(0).expand(batch_t, -1, -1, -1)  # (batch_t, h, w, 2)
    deformed_pixel_coords = pixel_grid_expanded + shifts_px  # (batch_t, h, w, 2)
    
    if frames.is_cuda:
        torch.cuda.synchronize()
    t5 = time.time()
    
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
    
    if frames.is_cuda:
        torch.cuda.synchronize()
    t6 = time.time()
    
    return corrected_frames

def correct_motion_fast(
    image: torch.Tensor,
    pixel_spacing: float,
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
    pixel_spacing: float
        Pixel spacing in angstroms
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
                       f"Final two dimensions must be (1, 1) for single patch correction.")
    
    t, h, w = image.shape
    
    # Extract shifts from deformation field (2, t, 1, 1) -> (t, 2)
    shifts = einops.rearrange(deformation_grid, 'c t 1 1 -> t c')
    shifts *= -1 # flip for phase shift

    print(f"Single patch correction: applying shifts to {t} frames")
    print(f"Shift range: y=[{shifts[:, 0].min():.2f}, {shifts[:, 0].max():.2f}], "
          f"x=[{shifts[:, 1].min():.2f}, {shifts[:, 1].max():.2f}] pixels")
    
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
    '''
    corrected_frames = image.clone()
    for i in range(t):
        if abs(shifts[i,0]) > 1e-6 or abs(shifts[i,1]) > 1e-6:
            corrected_frames[i] = apply_phase_shift(image[i], shifts[i,1].item(), shifts[i,0].item())
    '''
    return corrected_frames


def apply_phase_shift(image: torch.Tensor, shift_x: float, shift_y: float) -> torch.Tensor:
    """Apply phase shift in Fourier space."""
    if abs(shift_x) < 1e-6 and abs(shift_y) < 1e-6:
        return image
    
    h, w = image.shape
    
    # Create frequency grids
    freq_y = torch.fft.fftfreq(h, device=image.device)
    freq_x = torch.fft.fftfreq(w, device=image.device)
    
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    
    # Phase shift: exp(-2πi * (fx*shift_x + fy*shift_y))
    phase_shift = torch.exp(-2j * torch.pi * (fx * shift_x + fy * shift_y))
    
    # Apply shift in Fourier space
    image_fft = torch.fft.fft2(image)
    shifted_fft = image_fft * phase_shift
    shifted_image = torch.fft.ifft2(shifted_fft).real
    
    return shifted_image

