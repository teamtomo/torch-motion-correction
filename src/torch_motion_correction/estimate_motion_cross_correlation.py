import torch
import torch.nn.functional as F
import einops
import numpy as np
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.envelopes import b_envelope
from torch_grid_utils import circle
from scipy.signal import savgol_filter
from torch_fourier_shift import fourier_shift_dft_2d

from torch_image_interpolation import sample_image_1d

from torch_motion_correction.patch_grid import patch_grid_lazy
from torch_motion_correction.utils import (
    spatial_frequency_to_fftfreq,
    normalize_image,
)
from torch_motion_correction.correct_motion import correct_motion, correct_motion_fast


def _apply_sub_pixel_refinement2(cross_corr: torch.Tensor, peak_indices: torch.Tensor, patch_height: int, patch_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply sub-pixel refinement to peak positions using parabolic fitting.
    
    Parameters
    ----------
    cross_corr: torch.Tensor
        Cross-correlation values (gh*gw, ph*pw)
    peak_indices: torch.Tensor
        Integer peak indices (gh*gw,)
    patch_height: int
        Height of patches
    patch_width: int
        Width of patches
        
    Returns
    -------
    peak_y: torch.Tensor
        Refined y coordinates (gh*gw,)
    peak_x: torch.Tensor
        Refined x coordinates (gh*gw,)
    """
    num_patches = cross_corr.shape[0]
    
    # Convert flat indices to 2D coordinates
    peak_y_int = peak_indices // patch_width
    peak_x_int = peak_indices % patch_width
    
    # Initialize refined coordinates
    peak_y_refined = peak_y_int.float()
    peak_x_refined = peak_x_int.float()
    
    # Reshape cross_corr to 3D for easier neighborhood access
    cross_corr_3d = cross_corr.view(num_patches, patch_height, patch_width)
    
    # Apply sub-pixel refinement to each patch using parabolic fitting
    for i in range(num_patches):
        y_int = peak_y_int[i].item()
        x_int = peak_x_int[i].item()
        
        # Check bounds for parabolic fit (need 3x3 neighborhood)
        if 1 <= y_int < patch_height - 1 and 1 <= x_int < patch_width - 1:
            # Extract 3x3 neighborhood around peak
            y_vals = cross_corr_3d[i, y_int-1:y_int+2, x_int]
            x_vals = cross_corr_3d[i, y_int, x_int-1:x_int+2]
            
            # Parabolic fit for y direction
            if y_vals[2] != y_vals[0]:
                y_offset = 0.5 * (y_vals[0] - y_vals[2]) / (y_vals[0] - 2*y_vals[1] + y_vals[2])
                peak_y_refined[i] += y_offset
            
            # Parabolic fit for x direction  
            if x_vals[2] != x_vals[0]:
                x_offset = 0.5 * (x_vals[0] - x_vals[2]) / (x_vals[0] - 2*x_vals[1] + x_vals[2])
                peak_x_refined[i] += x_offset
    
    return peak_y_refined, peak_x_refined


def _apply_temporal_smoothing(
    deformation_field: torch.Tensor, 
    window_size: int, 
    device: torch.device
) -> torch.Tensor:
    """
    Apply temporal smoothing to deformation field across frames for each patch.
    
    Parameters
    ----------
    deformation_field: torch.Tensor
        Deformation field (2, t, gh, gw)
    window_size: int
        Window size for smoothing (must be odd)
    device: torch.device
        Device for computation
        
    Returns
    -------
    smoothed_deformation_field: torch.Tensor
        Smoothed deformation field (2, t, gh, gw)
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    # Ensure window size doesn't exceed number of frames
    t = deformation_field.shape[1]
    window_size = min(window_size, t)
    
    if window_size < 3:
        return deformation_field  # No smoothing for very small windows
    
    # Apply Savitzky-Golay smoothing to each patch independently
    smoothed_field = deformation_field.clone()
    
    # Process each patch position
    for gy in range(deformation_field.shape[2]):
        for gx in range(deformation_field.shape[3]):
            # Extract time series for this patch
            y_series = deformation_field[0, :, gy, gx].cpu().numpy()
            x_series = deformation_field[1, :, gy, gx].cpu().numpy()
            
            # Apply Savitzky-Golay smoothing
            if len(y_series) >= window_size:
                smoothed_y = savgol_filter(y_series, window_size, 1)
                smoothed_x = savgol_filter(x_series, window_size, 1)
                
                # Convert back to torch and update
                smoothed_field[0, :, gy, gx] = torch.from_numpy(smoothed_y).to(device)
                smoothed_field[1, :, gy, gx] = torch.from_numpy(smoothed_x).to(device)
    
    return smoothed_field


def _update_deformation_field_dimensions(
    deformation_field: torch.Tensor, 
    target_shape: tuple[int, int, int, int], 
    device: torch.device
) -> torch.Tensor:
    """
    Update deformation field dimensions by filling with mean values from initial field.
    
    Parameters
    ----------
    deformation_field: torch.Tensor
        Existing deformation field (2, t_old, gh_old, gw_old)
    target_shape: tuple[int, int, int, int]
        Target shape (2, t_new, gh_new, gw_new)
    device: torch.device
        Device for computation
        
    Returns
    -------
    updated_deformation_field: torch.Tensor
        Updated deformation field with target dimensions filled with mean values
    """
    if deformation_field.shape == target_shape:
        return deformation_field.to(device)
    
    # Calculate mean values over patches for each frame
    y_means = torch.mean(deformation_field[0], dim=(1, 2))  # (t_old,)
    x_means = torch.mean(deformation_field[1], dim=(1, 2))  # (t_old,)
    
    # Get dimensions
    _, t_old, gh_old, gw_old = deformation_field.shape
    _, t_new, gh_new, gw_new = target_shape
    
    # Create new tensor with target shape
    updated_field = torch.zeros(target_shape, device=device, dtype=deformation_field.dtype)
    
    # Interpolate temporal means to new frame count
    if t_old > 1 and t_new > 1:
        # Create time coordinates for interpolation
        t_new_coords = torch.linspace(0, t_old - 1, steps=t_new, device=device)

        # Interpolate y and x means
        y_interp = sample_image_1d(y_means, t_new_coords, interpolation='linear')
        x_interp = sample_image_1d(x_means, t_new_coords, interpolation='linear')
    else:
        # If only one frame, use that value for all frames
        y_interp = y_means[0].expand(t_new)
        x_interp = x_means[0].expand(t_new)
    
    # Fill each frame with its interpolated mean values
    for t in range(t_new):
        updated_field[0, t, :, :] = y_interp[t]
        updated_field[1, t, :, :] = x_interp[t]
    
    print(f"Updated deformation field from {deformation_field.shape} to {target_shape}")
    print(f"Interpolated {t_old} frames to {t_new} frames")
    print(f"Y range: [{y_interp.min():.3f}, {y_interp.max():.3f}], X range: [{x_interp.min():.3f}, {x_interp.max():.3f}]")
    
    return updated_field


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

    #print all shifts
    #print(f"Shifts: {shifts}")
    
    # Create deformation field with constant shifts per frame
    # Shape: (2, t, 1, 1) - 2 channels (y, x), t time points, 1x1 spatial
    # Use einops to reshape and repeat the shifts tensor
    # shifts: (t, 2) -> (2, t, 1, 1)
    deformation_field = einops.rearrange(shifts, 't c -> c t 1 1')
    # (2, t, 1, 1) -> (2, t, 32, 32)
    #deformation_field = einops.repeat(deformation_field, 'c t 1 1 -> c t h w', h=32, w=32)
    

    
    return deformation_field

def estimate_motion_cross_correlation_whole_image(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    reference_frame: int = None,  # None for middle frame
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms
    device: torch.device = None,
) -> torch.Tensor:
    """
    Estimate motion using cross-correlation for the whole image.
    
    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    pixel_spacing: float
        Pixel spacing in angstroms
    reference_frame: int, optional
        Frame index to use as reference. If None, uses middle frame.
    b_factor: float
        B-factor for frequency filtering
    frequency_range: tuple[float, float]
        Frequency range for bandpass filtering in angstroms
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    shifts: torch.Tensor
        (t, 2) array of shifts for each frame in pixels (y, x)
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        
    t, h, w = image.shape
    
    # Use middle frame as reference if not specified
    if reference_frame is None:
        reference_frame = t // 2
        
    print(f"Cross-correlation whole image: using frame {reference_frame} as reference")
    
    # Normalize image
    image = normalize_image(image)
    
    # Apply circular mask to reduce edge artifacts
    mask = circle(
        radius=min(h, w) / 4,
        image_shape=(h, w),
        smoothing_radius=min(h, w) / 8,
        device=device
    )
    
    # Apply mask and FFT
    masked_images = image * mask
    fft_images = torch.fft.rfftn(masked_images, dim=(-2, -1))
    
    # Prepare filters
    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=(h, w),
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device
    )
    
    bandpass = _prepare_bandpass_filter(
        frequency_range=frequency_range,
        patch_shape=(h, w),
        pixel_spacing=pixel_spacing,
        device=device
    )
    
    # Apply filters
    filtered_fft = fft_images * bandpass * b_factor_envelope
    
    # Reference frame
    reference_fft = filtered_fft[reference_frame]
    
    # Calculate shifts for each frame
    shifts = torch.zeros((t, 2), device=device)  # (y, x) shifts
    
    for frame_idx in range(t):
        if frame_idx == reference_frame:
            continue  # Reference frame has zero shift
            
        # Cross-correlation in frequency domain
        frame_fft = filtered_fft[frame_idx]
        cross_corr_fft = torch.conj(reference_fft) * frame_fft
        cross_corr = torch.fft.irfftn(cross_corr_fft, s=(h, w))
        
        # Find peak position
        peak_idx = torch.argmax(cross_corr.flatten())
        peak_y, peak_x = divmod(peak_idx.item(), w)
        
        # Convert to shifts (handle wraparound)
        shift_y = peak_y if peak_y <= h // 2 else peak_y - h
        shift_x = peak_x if peak_x <= w // 2 else peak_x - w
        
        shifts[frame_idx] = torch.tensor([shift_y, shift_x], device=device)
    
    print(f"Estimated shifts range: y=[{shifts[:, 0].min():.1f}, {shifts[:, 0].max():.1f}], "
          f"x=[{shifts[:, 1].min():.1f}, {shifts[:, 1].max():.1f}]")
    
    final_deformation_field = create_deformation_field_from_whole_image_shifts(
        shifts=shifts,
        image_shape=(t, h, w),
        device=device
    )
    
    return final_deformation_field


def estimate_motion_cross_correlation_patches(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    reference_frame: int = None,  # None for middle frame
    reference_strategy: str = "mean_except_current",  # "middle_frame" or "mean_except_current"
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms
    patch_sidelength: int = 1024,
    sub_pixel_refinement: bool = True,  # Enable sub-pixel peak finding
    temporal_smoothing: bool = True,  # Enable temporal smoothing across frames
    smoothing_window_size: int = 5,  # Window size for temporal smoothing
    deformation_field: torch.Tensor = None,  # Optional deformation field to apply
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate motion using cross-correlation for the patches.
    
    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    pixel_spacing: float
        Pixel spacing in angstroms
    reference_frame: int, optional
        Frame index to use as reference. If None, uses middle frame.
    reference_strategy: str
        Strategy for reference frame selection:
        - "middle_frame": Use the middle frame as reference for all frames
        - "mean_except_current": Use mean of all frames except current frame as reference
    b_factor: float
        B-factor for frequency filtering
    frequency_range: tuple[float, float]
        Frequency range for bandpass filtering in angstroms
    patch_sidelength: int
        Size of patches (assumed square)
    sub_pixel_refinement: bool
        Whether to use sub-pixel peak finding for more accurate motion estimation
    temporal_smoothing: bool
        Whether to apply temporal smoothing across frames for each patch
    smoothing_window_size: int
        Window size for temporal smoothing (must be odd number)
    deformation_field: torch.Tensor, optional
        Optional deformation field to apply to the image before motion estimation.
        If provided, the image will be corrected using this field first, and the newly
        calculated motion will be added to this field (cumulative motion correction).
        If the last two dimensions are (1, 1), uses correct_motion_fast for single patch.
        If the last two dimensions match the patch grid dimensions (gh, gw), uses correct_motion_fast for patch grid.
        Otherwise, uses correct_motion for full deformation field.
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    deformation_field: torch.Tensor
        (2, t, gh, gw) deformation field for motion correction, where gh and gw are the patch grid dimensions
    data_patch_positions: torch.Tensor
        (t, gh, gw, 3) patch center positions
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        
    t, h, w = image.shape

    # Use middle frame as reference if not specified
    if reference_frame is None:
        reference_frame = t // 2
    
    image = normalize_image(image)
        
    if reference_strategy == "middle_frame":
        print(f"Cross-correlation patches: using frame {reference_frame} as reference")
    else:
        print(f"Cross-correlation patches: using mean of all frames except current frame as reference")

    # Apply deformation field if provided
    if deformation_field is not None:
        deformation_field = deformation_field.to(device)
        
        # Check if it's a single patch deformation field (last two dims are 1, 1)
        if deformation_field.shape[-2:] == (1, 1):
            print("Applying single patch deformation field using correct_motion_fast")
            image = correct_motion_fast(
                image=image,
                pixel_spacing=pixel_spacing,
                deformation_grid=deformation_field,
                device=device
            )
        else:
            print("Applying full deformation field using correct_motion")
            image = correct_motion(
                image=image,
                pixel_spacing=pixel_spacing,
                deformation_grid=deformation_field,
                device=device
            )
    # Split into patch grid size patches with 50% overlap
    ph, pw = patch_sidelength, patch_sidelength
    lazy_patch_grid, data_patch_positions = patch_grid_lazy(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),
        distribute_patches=True,
    )
    gh, gw = data_patch_positions.shape[1:3]
    print(f"Number of patches per frame: {gh * gw}")

    # Prepare filters (only need to do this once)
    mask = circle(
        radius=pw / 4,
        image_shape=(ph, pw),
        smoothing_radius=pw / 8,
        device=device
    )
    
    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=(ph, pw),
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device
    )

    bandpass = _prepare_bandpass_filter(
        frequency_range=frequency_range,
        patch_shape=(ph, pw),
        pixel_spacing=pixel_spacing,
        device=device
    )
    
    # Initialize or update deformation field
    if deformation_field is None:
        deformation_field = torch.zeros((2, t, gh, gw), device=device)
    else:
        # Update existing deformation field to match new dimensions

        deformation_field = _update_deformation_field_dimensions(
            deformation_field, (2, t, gh, gw), device
        )
        print(f"Using existing deformation field as base for cumulative motion correction")
    
    # Process each frame individually to manage memory
    for frame_idx in range(t):
        if reference_strategy == "middle_frame" and frame_idx == reference_frame:
            continue  # Reference frame has zero shift
            
        print(f"Processing frame {frame_idx}/{t-1}")
        
        # Determine reference patches based on strategy
        if reference_strategy == "middle_frame":
            # Use middle frame as reference
            ref_patches = lazy_patch_grid[reference_frame]  # (1, gh, gw, 1, ph, pw)
            ref_patches = einops.rearrange(ref_patches, '1 gh gw 1 ph pw -> gh gw ph pw')
        elif reference_strategy == "mean_except_current":
            # Use mean of all frames except current frame (computed incrementally to save memory)
            ref_patches = None
            count = 0
            for other_frame_idx in range(t):
                if other_frame_idx != frame_idx:
                    other_patches = lazy_patch_grid[other_frame_idx]  # (1, gh, gw, 1, ph, pw)
                    other_patches = einops.rearrange(other_patches, '1 gh gw 1 ph pw -> gh gw ph pw')
                    if ref_patches is None:
                        ref_patches = other_patches.clone()
                    else:
                        ref_patches += other_patches
                    count += 1
            ref_patches = ref_patches / count
        else:
            raise ValueError(f"Unknown reference_strategy: {reference_strategy}")
        
        # Get patches for current frame only
        frame_patches = lazy_patch_grid[frame_idx]  # (1, gh, gw, 1, ph, pw)
        frame_patches = einops.rearrange(frame_patches, '1 gh gw 1 ph pw -> gh gw ph pw')
        
        # Apply mask and filters to reference patches
        ref_patches *= mask
        ref_patches_fft = torch.fft.rfftn(ref_patches, dim=(-2, -1))
        ref_patches_fft = ref_patches_fft * bandpass * b_factor_envelope
        
        # Apply mask and filters to frame patches
        frame_patches *= mask
        frame_patches_fft = torch.fft.rfftn(frame_patches, dim=(-2, -1))
        frame_patches_fft = frame_patches_fft * bandpass * b_factor_envelope
        
        # Vectorized cross-correlation for all patches at once
        cross_corr_fft = torch.conj(ref_patches_fft) * frame_patches_fft
        cross_corr = torch.fft.irfftn(cross_corr_fft, s=(ph, pw))
        
        # Vectorized peak finding for all patches
        # Reshape to (gh*gw, ph*pw) for vectorized argmax
        cross_corr_flat = cross_corr.view(gh * gw, ph * pw)
        peak_indices = torch.argmax(cross_corr_flat, dim=1)  # (gh*gw,)
        
        if sub_pixel_refinement:
            # Use sub-pixel peak finding
            peak_y, peak_x = _apply_sub_pixel_refinement2(cross_corr_flat, peak_indices, ph, pw)
        else:
            # Use integer peak finding
            peak_y = peak_indices // pw
            peak_x = peak_indices % pw
        
        # Convert to shifts (handle wraparound)
        shift_y = torch.where(peak_y <= ph // 2, peak_y, peak_y - ph)
        shift_x = torch.where(peak_x <= pw // 2, peak_x, peak_x - pw)
        
        # Reshape back to grid
        shift_y = shift_y.view(gh, gw)
        shift_x = shift_x.view(gh, gw)
        
        
        # Add shifts to existing deformation field (cumulative motion correction)
        deformation_field[0, frame_idx, :, :] += shift_y
        deformation_field[1, frame_idx, :, :] += shift_x

    # Apply temporal smoothing if enabled
    if temporal_smoothing:
        print(f"Applying temporal smoothing with window size {smoothing_window_size}")
        deformation_field = _apply_temporal_smoothing(
            deformation_field, 
            smoothing_window_size, 
            device
        )
        print(f"After temporal smoothing - range: y=[{deformation_field[0].min():.1f}, {deformation_field[0].max():.1f}], "
              f"x=[{deformation_field[1].min():.1f}, {deformation_field[1].max():.1f}]")

    print(f"Estimated deformation field range: y=[{deformation_field[0].min():.1f}, {deformation_field[0].max():.1f}], "
          f"x=[{deformation_field[1].min():.1f}, {deformation_field[1].max():.1f}]")
    
    print(f"Estimated deformation field shape: {deformation_field.shape}")
    return deformation_field, data_patch_positions


def refine_correlation_patches(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    reference_frame: int = None,  # None for middle frame
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms
    patch_sidelength: int = 1024,
    max_iterations: int = 20,
    inner_radius_for_peak_search: float = 0.0,
    outer_radius_for_peak_search: float = 50.0,
    max_shift_convergence_threshold: float = 0.01,
    savitzky_golay_window_size: int = 5,
    sub_pixel_refinement: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Refine motion estimation using cross-correlation for patches with advanced features.
    
    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    pixel_spacing: float
        Pixel spacing in angstroms
    reference_frame: int, optional
        Frame index to use as reference. If None, uses middle frame.
    b_factor: float
        B-factor for frequency filtering
    frequency_range: tuple[float, float]
        Frequency range for bandpass filtering in angstroms
    patch_sidelength: int
        Size of patches (assumed square)
    max_iterations: int
        Maximum number of refinement iterations
    inner_radius_for_peak_search: float
        Inner radius for peak search (pixels)
    outer_radius_for_peak_search: float
        Outer radius for peak search (pixels)
    max_shift_convergence_threshold: float
        Convergence threshold for shifts (pixels)
    savitzky_golay_window_size: int
        Window size for Savitzky-Golay smoothing
    sub_pixel_refinement: bool
        Whether to apply sub-pixel refinement
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    deformation_field: torch.Tensor
        (2, t, h, w) deformation field for motion correction
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        
    t, h, w = image.shape

    # Use middle frame as reference if not specified
    if reference_frame is None:
        reference_frame = t // 2
    
    image = normalize_image(image)
        
    print(f"Refining patch-based correlation: using frame {reference_frame} as reference")
    print(f"Max iterations: {max_iterations}, convergence threshold: {max_shift_convergence_threshold}")
    
    # Split into patch grid size patches with 50% overlap
    ph, pw = patch_sidelength, patch_sidelength
    lazy_patch_grid, data_patch_positions = patch_grid_lazy(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),
        distribute_patches=True,
    )

    gh, gw = data_patch_positions.shape[1:3]
    print(f"Number of patches per frame: {gh * gw}")

    # Prepare filters (only need to do this once)
    mask = circle(
        radius=pw / 4,
        image_shape=(ph, pw),
        smoothing_radius=pw / 8,
        device=device
    )
    
    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=(ph, pw),
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device
    )

    bandpass = _prepare_bandpass_filter(
        frequency_range=frequency_range,
        patch_shape=(ph, pw),
        pixel_spacing=pixel_spacing,
        device=device
    )
    
    # Initialize deformation field
    deformation_field = torch.zeros((2, t, gh, gw), device=device)
    
    # Initialize shift tracking for refinement
    total_shifts = torch.zeros((t, 2, gh, gw), device=device, dtype=torch.float32)  # (y, x) shifts per frame
    
    # Iterative refinement
    for iteration in range(max_iterations):
        print(f"Refinement iteration {iteration + 1}/{max_iterations}")
        
        # Store current iteration shifts
        current_shifts = torch.zeros((t, 2, gh, gw), device=device, dtype=torch.float32)  # (y, x) shifts
        
        # Process each frame individually to manage memory
        for frame_idx in range(t):
            #print(f"Processing frame {frame_idx}/{t-1}")
            
            # Get patches for current frame only
            frame_patches = lazy_patch_grid[frame_idx]  # (1, gh, gw, 1, ph, pw)
            #if frame_idx == 0:
                #print(f"frame_patches shape: {frame_patches.shape}")
                # Print the patch centers for the current frame
                #print(f"Patch centers for frame {frame_idx}: {data_patch_positions[frame_idx]}")
            frame_patches = einops.rearrange(frame_patches, '1 gh gw 1 ph pw -> gh gw ph pw')
            
            # Apply mask and filters
            frame_patches *= mask
            frame_patches_fft = torch.fft.rfftn(frame_patches, dim=(-2, -1))
            frame_patches_fft = frame_patches_fft * bandpass * b_factor_envelope
            
            # Apply shifts to patches
            shifted_patches = fourier_shift_dft_2d(
                dft=frame_patches_fft,
                image_shape=(ph, pw),
                shifts=-1*einops.rearrange(total_shifts[frame_idx], 'c gh gw -> gh gw c'),
                rfft=True,
                fftshifted=False,
            )
            
            # Create reference patches as sum of all other frames (sum minus current)
            # We need to accumulate patches from all other frames
            ref_patches_fft = None
            for other_frame_idx in range(t):
                if other_frame_idx == frame_idx:
                    continue  # Skip current frame
                    
                # Get patches for other frame
                other_patches = lazy_patch_grid[other_frame_idx]  # (1, gh, gw, 1, ph, pw)
                other_patches = einops.rearrange(other_patches, '1 gh gw 1 ph pw -> gh gw ph pw')
                
                # Apply mask and filters
                other_patches *= mask
                other_patches_fft = torch.fft.rfftn(other_patches, dim=(-2, -1))
                other_patches_fft = other_patches_fft * bandpass * b_factor_envelope
                
                # Apply shifts to other frame patches
                other_shifted_patches = fourier_shift_dft_2d(
                    dft=other_patches_fft,
                    image_shape=(ph, pw),
                    shifts=-1*einops.rearrange(total_shifts[other_frame_idx], 'c gh gw -> gh gw c'),
                    rfft=True,
                    fftshifted=False,
                )
                
                # Accumulate reference patches
                if ref_patches_fft is None:
                    ref_patches_fft = other_shifted_patches
                else:
                    ref_patches_fft += other_shifted_patches
            
            # Average the reference patches
            ref_patches_fft = ref_patches_fft / (t - 1)
            
            # Vectorized cross-correlation for all patches at once
            cross_corr_fft = torch.conj(ref_patches_fft) * shifted_patches
            cross_corr = torch.fft.irfftn(cross_corr_fft, s=(ph, pw))
            
            # Apply radius constraints for peak search
            if outer_radius_for_peak_search > 0 or inner_radius_for_peak_search > 0:
                cross_corr = _apply_radius_constraints(
                    cross_corr, 
                    inner_radius_for_peak_search, 
                    outer_radius_for_peak_search
                )
            
            # Vectorized peak finding for all patches
            # Reshape to (gh*gw, ph*pw) for vectorized argmax
            cross_corr_flat = cross_corr.view(gh * gw, ph * pw)
            peak_indices = torch.argmax(cross_corr_flat, dim=1)  # (gh*gw,)
            
            # Convert to 2D coordinates
            peak_y = peak_indices // pw
            peak_x = peak_indices % pw
            
            # Convert to shifts (handle wraparound)
            shift_y = torch.where(peak_y <= ph // 2, peak_y, peak_y - ph)
            shift_x = torch.where(peak_x <= pw // 2, peak_x, peak_x - pw)
            
            # Apply sub-pixel refinement if requested
            if sub_pixel_refinement:
                shift_y, shift_x = _apply_sub_pixel_refinement(
                    cross_corr, peak_y, peak_x, shift_y, shift_x, gh, gw
                )
            
            # Reshape back to grid
            shift_y = shift_y.view(gh, gw)
            shift_x = shift_x.view(gh, gw)
            
            current_shifts[frame_idx, 0] = shift_y
            current_shifts[frame_idx, 1] = shift_x
        
        # Smooth the shifts using Savitzky-Golay filter
        if savitzky_golay_window_size < t and savitzky_golay_window_size >= 3:
            # Ensure odd window size
            window_size = savitzky_golay_window_size
            if window_size % 2 == 0:
                window_size += 1
            
            # Apply smoothing to each patch position across time
            for gy in range(gh):
                for gx in range(gw):
                    # Convert to numpy for scipy
                    current_np_y = current_shifts[:, 0, gy, gx].cpu().numpy()
                    current_np_x = current_shifts[:, 1, gy, gx].cpu().numpy()
                    
                    # Apply Savitzky-Golay smoothing
                    smoothed_shifts_y = savgol_filter(current_np_y, window_size, 1)
                    smoothed_shifts_x = savgol_filter(current_np_x, window_size, 1)
                    
                    # Convert back to torch
                    smoothed_shifts_y = torch.from_numpy(smoothed_shifts_y).to(device)
                    smoothed_shifts_x = torch.from_numpy(smoothed_shifts_x).to(device)
                    
                    # Update current shifts
                    current_shifts[:, 0, gy, gx] = smoothed_shifts_y
                    current_shifts[:, 1, gy, gx] = smoothed_shifts_x
        
        # Center shifts around middle frame
        middle_frame = t // 2
        middle_shift = current_shifts[middle_frame].clone()  # (2, gh, gw)
        current_shifts = current_shifts - middle_shift.unsqueeze(0)  # Broadcast across time
        
        # Check convergence
        max_shift = torch.sqrt(current_shifts[:, 0]**2 + current_shifts[:, 1]**2).max().item()
        print(f"Iteration {iteration + 1}: max shift = {max_shift:.3f} pixels")
        
        # Update total shifts
        total_shifts += current_shifts
        
        # Check convergence
        if max_shift <= max_shift_convergence_threshold:
            print(f"Converged after {iteration + 1} iterations (max_shift = {max_shift:.3f})")
            break

    # Update deformation field
    # total_shifts shape: (t, 2, gh, gw)
    # deformation_field shape: (2, t, gh, gw)
    deformation_field = einops.rearrange(total_shifts, 't c gh gw -> c t gh gw') 

    print(f"Final deformation field range: y=[{deformation_field[0].min():.1f}, {deformation_field[0].max():.1f}], "
          f"x=[{deformation_field[1].min():.1f}, {deformation_field[1].max():.1f}]")
    
    return deformation_field


def _apply_radius_constraints(
    cross_corr: torch.Tensor, 
    inner_radius: float, 
    outer_radius: float
) -> torch.Tensor:
    """Apply radius constraints to cross-correlation for peak search."""
    m, n, h, w = cross_corr.shape
    center_y, center_x = h // 2, w // 2
    
    # Create radius mask
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=cross_corr.device),
        torch.arange(w, device=cross_corr.device),
        indexing='ij'
    )
    
    distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # Apply radius constraints
    if outer_radius > 0:
        mask = distances <= outer_radius
        if inner_radius > 0:
            mask &= distances >= inner_radius
        
        # Mask the cross-correlation
        masked_cross_corr = cross_corr.clone()
        mask_broadcast = einops.repeat(mask, 'h w -> m n h w', m=m, n=n)
        masked_cross_corr[~mask_broadcast] = cross_corr.min()
        return masked_cross_corr
    else:
        return cross_corr


def _apply_sub_pixel_refinement(
    cross_corr: torch.Tensor,
    peak_y: torch.Tensor,
    peak_x: torch.Tensor,
    shift_y: torch.Tensor,
    shift_x: torch.Tensor,
    gh: int,
    gw: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply sub-pixel refinement using parabolic fitting."""
    # Handle 4D cross_corr tensor (gh, gw, ph, pw)
    if len(cross_corr.shape) == 4:
        gh_corr, gw_corr, h, w = cross_corr.shape
        # Reshape to (gh*gw, ph, pw) for vectorized operations
        cross_corr_flat = cross_corr.view(gh_corr * gw_corr, h, w)
    else:
        # Handle 2D case (ph, pw)
        h, w = cross_corr.shape
        cross_corr_flat = cross_corr.unsqueeze(0)  # Add batch dimension
    
    center_y, center_x = h // 2, w // 2
    
    # Convert to float to allow sub-pixel refinement
    refined_shift_y = shift_y.clone().float()
    refined_shift_x = shift_x.clone().float()
    
    for i in range(gh * gw):
        py, px = peak_y[i].item(), peak_x[i].item()
        
        # Check bounds for parabolic fit
        if 1 <= py < h - 1 and 1 <= px < w - 1:
            # Fit parabola around peak for sub-pixel accuracy
            y_vals = cross_corr_flat[i, py-1:py+2, px]
            x_vals = cross_corr_flat[i, py, px-1:px+2]
            
            # Parabolic fit for y direction
            if y_vals[2] != y_vals[0]:
                y_offset = 0.5 * (y_vals[0] - y_vals[2]) / (y_vals[0] - 2*y_vals[1] + y_vals[2])
                refined_shift_y[i] += y_offset
            
            # Parabolic fit for x direction  
            if x_vals[2] != x_vals[0]:
                x_offset = 0.5 * (x_vals[0] - x_vals[2]) / (x_vals[0] - 2*x_vals[1] + x_vals[2])
                refined_shift_x[i] += x_offset
    
    return refined_shift_y, refined_shift_x


def _prepare_bandpass_filter(
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
