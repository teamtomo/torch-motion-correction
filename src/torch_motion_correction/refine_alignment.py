"""
Cross-correlation based alignment refinement.

This module implements a refinement step similar to the unblur algorithm,
where each frame is aligned to the sum of all other frames using cross-correlation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from scipy.signal import savgol_filter
import einops

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


def refine_alignment(
    images: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,
    max_iterations: int = 10,
    b_factor: float = 500.0,
    mask_central_cross: bool = True,
    cross_mask_width: int = 5,
    inner_radius_for_peak_search: float = 0.0,
    outer_radius_for_peak_search: float = 50.0,
    max_shift_convergence_threshold: float = 0.1,
    number_of_frames_for_running_average: int = 1,
    savitzky_golay_window_size: int = 5,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Refine alignment using cross-correlation with sum-minus-current reference.
    
    Args:
        images: Input image stack (t, h, w)
        pixel_spacing: Pixel spacing in angstroms
        max_iterations: Maximum number of refinement iterations
        b_factor: B-factor for frequency domain filtering
        mask_central_cross: Whether to mask the central cross in Fourier space
        cross_mask_width: Width of the cross mask
        inner_radius_for_peak_search: Inner radius for peak search (pixels)
        outer_radius_for_peak_search: Outer radius for peak search (pixels)
        max_shift_convergence_threshold: Convergence threshold for shifts (pixels)
        number_of_frames_for_running_average: Number of frames for running average
        savitzky_golay_window_size: Window size for Savitzky-Golay smoothing
        device: Device to run computations on
        
    Returns:
        Tuple of (aligned_images, x_shifts, y_shifts)
    """
    if device is None:
        device = images.device
    else:
        images = images.to(device)
    
    t, h, w = images.shape
    
    # Initialize shift arrays
    shifts = torch.zeros((t, 2), device=device)  # (y, x) shifts   
    # Make a copy of images for modification
    aligned_images = images.clone()
    
    # Running average parameters
    running_average_half_size = max(1, (number_of_frames_for_running_average - 1) // 2)
    
    print(f"Starting alignment refinement with {max_iterations} max iterations")
    
    for iteration in range(1, max_iterations + 1):
        print(f"Refinement iteration {iteration}/{max_iterations}")
        
        # Compute sum of all images
        sum_of_images = torch.sum(aligned_images, dim=0)  # (h, w)
        
        # Create running averages if requested
        if number_of_frames_for_running_average > 1:
            stack_for_alignment = create_running_averages(
                aligned_images, running_average_half_size
            )
        else:
            stack_for_alignment = aligned_images
        
        # Store current iteration shifts
        current_shifts = torch.zeros((t, 2), device=device)  # (y, x) shifts
        
        # Align each frame to sum-minus-current
        for i in range(t):
            # Create reference by subtracting current frame from sum
            reference = sum_of_images - stack_for_alignment[i]
            
            # Apply B-factor filtering
            reference_filtered = apply_b_factor_filter(reference, b_factor, pixel_spacing)
            current_filtered = apply_b_factor_filter(stack_for_alignment[i], b_factor, pixel_spacing)
            
            # Mask central cross if requested
            if mask_central_cross:
                reference_filtered = mask_central_cross_fourier(reference_filtered, cross_mask_width)
                current_filtered = mask_central_cross_fourier(current_filtered, cross_mask_width)
            
            # Compute cross-correlation
            shift = find_shift_cross_correlation(
                reference_filtered, 
                current_filtered,
                inner_radius_for_peak_search,
                outer_radius_for_peak_search
            )
            
            current_shifts[i] = shift
        
        # Smooth the shifts
        total_shifts = shifts + current_shifts
        
        # Apply smoothing
        if savitzky_golay_window_size < t and savitzky_golay_window_size >= 3:
            # Ensure odd window size
            window_size = savitzky_golay_window_size
            if window_size % 2 == 0:
                window_size += 1
            
            # Convert to numpy for scipy
            total_np = total_shifts.cpu().numpy()

            
            # Apply Savitzky-Golay smoothing
            smoothed_shifts_y = savgol_filter(total_np[:,0], window_size, 1)
            smoothed_shifts_x = savgol_filter(total_np[:,1], window_size, 1)
            
            # Convert back to torch
            smoothed_shifts_y = torch.from_numpy(smoothed_shifts_y).to(device)
            smoothed_shifts_x = torch.from_numpy(smoothed_shifts_x).to(device)
            
            # Update current shifts
            current_shifts = torch.stack([smoothed_shifts_y, smoothed_shifts_x], dim=1) - shifts
        
        # Center shifts around middle frame
        middle_frame = t // 2
        middle_shift = current_shifts[middle_frame].clone()
        
        current_shifts = current_shifts - middle_shift
        
        # Check convergence
        max_shift = torch.sqrt(current_shifts[:,0]**2 + current_shifts[:,1]**2).max().item()
        print(f"Iteration {iteration}: max shift = {max_shift:.3f} pixels")
        
        # Apply shifts to images
        for i in range(t):
            if abs(current_shifts[i,0]) > 1e-6 or abs(current_shifts[i,1]) > 1e-6:
                aligned_images[i] = apply_phase_shift(
                    aligned_images[i], 
                    current_shifts[i,1].item(), 
                    current_shifts[i,0].item()
                )
        
        # Update total shifts
        shifts += current_shifts
        
        # Check convergence
        if max_shift <= max_shift_convergence_threshold:
            print(f"Converged after {iteration} iterations (max_shift = {max_shift:.3f})")
            break

    deformation_field = create_deformation_field_from_whole_image_shifts(
        shifts=shifts,
        image_shape=(t, h, w),
        device=device
    )
    
    return aligned_images, deformation_field


def create_running_averages(images: torch.Tensor, half_size: int) -> torch.Tensor:
    """Create running averages of nearby frames."""
    t, h, w = images.shape
    running_averages = torch.zeros_like(images)
    
    for i in range(t):
        start_frame = max(0, i - half_size)
        end_frame = min(t - 1, i + half_size)
        
        # Adjust window if we're at the edges
        if start_frame == 0:
            end_frame = min(t - 1, 2 * half_size)
        if end_frame == t - 1:
            start_frame = max(0, t - 1 - 2 * half_size)
        
        # Compute average
        running_averages[i] = torch.mean(images[start_frame:end_frame + 1], dim=0)
    
    return running_averages


def apply_b_factor_filter(image: torch.Tensor, b_factor: float, pixel_spacing: float) -> torch.Tensor:
    """Apply B-factor filtering in Fourier space."""
    h, w = image.shape
    
    # Create frequency grids
    freq_y = torch.fft.fftfreq(h, d=pixel_spacing, device=image.device)
    freq_x = torch.fft.fftfreq(w, d=pixel_spacing, device=image.device)
    
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_sq = fx**2 + fy**2
    
    # B-factor filter: exp(-B * freq^2 / 4)
    b_filter = torch.exp(-b_factor * freq_sq / 4.0)
    
    # Apply filter in Fourier space
    image_fft = torch.fft.fft2(image)
    filtered_fft = image_fft * b_filter
    filtered_image = torch.fft.ifft2(filtered_fft).real
    
    return filtered_image


def mask_central_cross_fourier(image: torch.Tensor, width: int) -> torch.Tensor:
    """Mask central cross in Fourier space."""
    h, w = image.shape
    
    # Transform to Fourier space
    image_fft = torch.fft.fft2(image)
    
    # Create mask
    mask = torch.ones_like(image_fft)
    
    # Mask vertical line
    center_x = w // 2
    mask[:, center_x - width//2:center_x + width//2 + 1] = 0
    
    # Mask horizontal line
    center_y = h // 2
    mask[center_y - width//2:center_y + width//2 + 1, :] = 0
    
    # Apply mask and transform back
    masked_fft = image_fft * mask
    masked_image = torch.fft.ifft2(masked_fft).real
    
    return masked_image


def find_shift_cross_correlation(
    reference: torch.Tensor,
    image: torch.Tensor,
    inner_radius: float,
    outer_radius: float
) -> Tuple[float, float]:
    """Find shift using cross-correlation and peak finding."""
    # Compute cross-correlation
    ref_fft = torch.fft.fft2(reference)
    img_fft = torch.fft.fft2(image)
    
    # Cross-correlation in Fourier space
    cross_corr_fft = ref_fft * torch.conj(img_fft)
    cross_corr = torch.fft.ifft2(cross_corr_fft).real
    
    # Shift zero frequency to center
    cross_corr = torch.fft.fftshift(cross_corr)
    
    # Find peak with radius constraints
    h, w = cross_corr.shape
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
        masked_cross_corr[~mask] = cross_corr.min()
    else:
        masked_cross_corr = cross_corr
    
    # Find peak
    peak_idx = torch.argmax(masked_cross_corr.flatten())
    peak_y, peak_x = divmod(peak_idx.item(), w)
    
    # Convert to shift (relative to center)
    shift_y = peak_y - center_y
    shift_x = peak_x - center_x
    
    # Parabolic sub-pixel refinement
    if 1 <= peak_y < h - 1 and 1 <= peak_x < w - 1:
        # Fit parabola around peak for sub-pixel accuracy
        y_vals = masked_cross_corr[peak_y-1:peak_y+2, peak_x]
        x_vals = masked_cross_corr[peak_y, peak_x-1:peak_x+2]
        
        # Parabolic fit for y direction
        if y_vals[2] != y_vals[0]:
            y_offset = 0.5 * (y_vals[0] - y_vals[2]) / (y_vals[0] - 2*y_vals[1] + y_vals[2])
            shift_y += y_offset
        
        # Parabolic fit for x direction  
        if x_vals[2] != x_vals[0]:
            x_offset = 0.5 * (x_vals[0] - x_vals[2]) / (x_vals[0] - 2*x_vals[1] + x_vals[2])
            shift_x += x_offset
    
    return torch.tensor([shift_y, shift_x], device=reference.device)


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
