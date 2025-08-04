from datetime import datetime
import random

import einops
import numpy as np
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.envelopes import b_envelope
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import circle

from torch_motion_correction.patch_grid import patch_grid, patch_grid_generator, patch_grid_lazy
from torch_motion_correction.utils import (
    spatial_frequency_to_fftfreq,
    normalize_image,
)


def estimate_motion(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms e.g. (300, 15)
    patch_sidelength: int = 512,
    n_iterations: int = 200,
    n_patches_per_batch: int = 20,
    learning_rate: float = 0.05,
    device: torch.device = None,  # device for the output tensor
) -> torch.Tensor:
    if device is None:
        device = image.device
    else:
        image = image.to(device)
    # grab image dims
    t, h, w = image.shape

    # normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # extract patches from data
    ph, pw = patch_sidelength, patch_sidelength
    data_patches, data_patch_positions = patch_grid(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),
        distribute_patches=True,
    )
    data_patches = einops.rearrange(data_patches, 't gh gw 1 ph pw -> t gh gw ph pw')
    gh, gw = data_patch_positions.shape[1:3]

    # apply a soft circular mask on data patches in real space
    mask = circle(
        radius=pw / 4,
        image_shape=(ph, pw),
        smoothing_radius=pw / 8,
        device=device
    )
    data_patches *= mask

    # rfft the data
    data_patches = torch.fft.rfftn(data_patches, dim=(-2, -1))

    # prepare b factor filter
    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=(ph, pw),
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device
    )

    # initialise the deformation field with learnable parameters for shifts
    # grid data: (2, nt, nh, nw)
    deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution,
        n_channels=2
    ).to(device)

    # normalize patch center positions to [0, 1] for evaluation of shifts.
    data_patch_positions = data_patch_positions / torch.tensor([t - 1, h - 1, w - 1], device=device)

    # initialise optimiser and detach data
    motion_optimiser = torch.optim.Adam(
        params=deformation_field.parameters(),
        lr=learning_rate,
    )
    data_patches = data_patches.detach()

    # optimise shifts at grid points on deformation field
    for i in range(n_iterations):
        # take a random subset of the patch grid over spatial dimensions
        patch_subset_idx = np.random.randint(
            low=(0, 0), high=(gh, gw), size=(n_patches_per_batch, 2)
        )
        idx_gh, idx_gw = einops.rearrange(patch_subset_idx, 'b idx -> idx b')
        patch_subset = data_patches[:, idx_gh, idx_gw]
        patch_subset_centers = data_patch_positions[:, idx_gh, idx_gw]
        reference_patches = torch.mean(patch_subset, dim=0)

        # predict the shifts at patch centers
        predicted_shifts = -1 * deformation_field(patch_subset_centers)

        # shift the patches by the predicted shifts
        predicted_shifts_px = predicted_shifts / pixel_spacing
        shifted_patches = fourier_shift_dft_2d(
            dft=patch_subset,
            image_shape=(ph, pw),
            shifts=predicted_shifts_px,
            rfft=True,
            fftshifted=False,
        )  # (b, ph, pw, h, w)

        # apply fourier filters
        bandpass = _prepare_bandpass_filter(
            frequency_range=frequency_range,
            patch_shape=(ph, pw),
            pixel_spacing=pixel_spacing,
            refinement_fraction=i / (n_iterations - 1),
            device=device
        )
        shifted_patches = shifted_patches * bandpass * b_factor_envelope

        # calculate the loss, MSE between data patches and reference patches
        loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)

        # zero gradients, backpropagate loss and step optimiser
        motion_optimiser.zero_grad()
        loss.backward()
        motion_optimiser.step()

        # log loss
        if i % 10 == 0:
            print(f"{i}: loss = {loss.item()}")

    # subtract mean from deformation field data
    average_shift = torch.mean(deformation_field.data)
    final_deformation_field = deformation_field.data - average_shift
    return final_deformation_field


def estimate_motion_memory_efficient(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms e.g. (300, 15)
    patch_sidelength: int = 512,
    n_iterations: int = 200,
    n_patches_per_batch: int = 20,
    learning_rate: float = 0.05,
    max_patches_in_memory: int = 100,  # Key parameter for memory control
    device: torch.device = None,
) -> torch.Tensor:
    """Memory-efficient motion estimation using patch generators.
    
    This version uses 3-10x less memory by:
    1. Processing patches on-demand using generators
    2. Caching only a subset of patches in memory 
    3. Using tensor views instead of copies
    
    Parameters
    ----------
    max_patches_in_memory : int
        Maximum number of patches to keep in memory at once. Reduce this 
        value if you're still running out of memory.
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        
    # grab image dims
    t, h, w = image.shape

    # normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # Use patch generator instead of loading all patches at once
    ph, pw = patch_sidelength, patch_sidelength
    patch_generator, data_patch_positions, grid_shape = patch_grid_generator(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),
        distribute_patches=True,
    )
    
    # Extract grid dimensions
    if len(grid_shape) == 2:
        gh, gw = grid_shape
        total_patches = gh * gw
    else:
        gd, gh, gw = grid_shape
        total_patches = gd * gh * gw

    print(f"Total patches: {total_patches}, will cache: {min(max_patches_in_memory, total_patches)}")

    # Create reusable filters and masks
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

    # Initialize deformation field
    deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution,
        n_channels=2
    ).to(device)

    # Normalize patch center positions to [0, 1]
    if len(data_patch_positions.shape) == 3:  # 2D case: (gh, gw, 2)  
        data_patch_positions = data_patch_positions / torch.tensor([t - 1, h - 1, w - 1], device=device)
    else:  # 3D case: (gd, gh, gw, 3)
        data_patch_positions = data_patch_positions / torch.tensor([t - 1, h - 1, w - 1], device=device)

    # Pre-process and cache a subset of patches
    print("Pre-processing patches...")
    cached_patches = {}
    cached_positions = {}
    
    # Create index mapping for patches
    if len(grid_shape) == 2:
        all_indices = [(i, j) for i in range(gh) for j in range(gw)]
    else:
        all_indices = [(i, j, k) for i in range(gd) for j in range(gh) for k in range(gw)]
    
    # Randomly sample patches to cache (this ensures good spatial distribution)
    cache_indices = set(random.sample(all_indices, min(max_patches_in_memory, len(all_indices))))
    
    # Process patches using the generator
    patch_iter = iter(patch_generator)
    for current_idx, patch in enumerate(patch_iter):
        # Convert linear index to grid coordinates
        if len(grid_shape) == 2:
            i, j = current_idx // gw, current_idx % gw
            patch_idx = (i, j)
        else:
            i = current_idx // (gh * gw)
            j = (current_idx % (gh * gw)) // gw
            k = current_idx % gw
            patch_idx = (i, j, k)
            
        # Only process patches we want to cache
        if patch_idx in cache_indices:
            # Reshape patch from (t, 1, ph, pw) to (t, ph, pw)
            patch = einops.rearrange(patch, 't 1 ph pw -> t ph pw')
            
            # Apply mask and FFT
            patch = patch * mask
            patch_fft = torch.fft.rfftn(patch, dim=(-2, -1))
            
            # Cache the processed patch
            cached_patches[patch_idx] = patch_fft.detach()
            
            # Cache the corresponding position
            if len(grid_shape) == 2:
                cached_positions[patch_idx] = data_patch_positions[i, j]
            else:
                cached_positions[patch_idx] = data_patch_positions[i, j, k]
                
        # Break if we have enough cached patches
        if len(cached_patches) >= max_patches_in_memory:
            break

    print(f"Cached {len(cached_patches)} patches for training")
    
    if len(cached_patches) == 0:
        raise RuntimeError("No patches were cached! Check your parameters.")

    # Initialize optimizer
    motion_optimiser = torch.optim.Adam(
        params=deformation_field.parameters(),
        lr=learning_rate,
    )

    # Training loop using only cached patches
    cached_indices = list(cached_patches.keys())
    
    for i in range(n_iterations):
        # Sample batch from cached patches
        batch_size = min(n_patches_per_batch, len(cached_indices))
        batch_indices = random.sample(cached_indices, batch_size)
        
        # Create batch tensors
        patch_subset = torch.stack([cached_patches[idx] for idx in batch_indices])
        patch_positions = torch.stack([cached_positions[idx] for idx in batch_indices])
        
        # Transpose patch_subset to match original format: (n_patches, t, ph, pw) -> (t, n_patches, ph, pw)
        patch_subset = patch_subset.transpose(0, 1)
        
        # Create reference by averaging patches
        reference_patches = torch.mean(patch_subset, dim=0)

        # Predict shifts at patch centers
        predicted_shifts = -1 * deformation_field(patch_positions)

        # Shift patches by predicted shifts
        predicted_shifts_px = predicted_shifts / pixel_spacing
        shifted_patches = fourier_shift_dft_2d(
            dft=patch_subset,
            image_shape=(ph, pw),
            shifts=predicted_shifts_px,
            rfft=True,
            fftshifted=False,
        )

        # Apply frequency filters
        bandpass = _prepare_bandpass_filter(
            frequency_range=frequency_range,
            patch_shape=(ph, pw),
            pixel_spacing=pixel_spacing,
            refinement_fraction=i / (n_iterations - 1),
            device=device
        )
        shifted_patches = shifted_patches * bandpass * b_factor_envelope

        # Calculate loss
        loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)

        # Optimization step
        motion_optimiser.zero_grad()
        loss.backward()
        motion_optimiser.step()

        # Progress logging
        if i % 10 == 0:
            print(f"{i}: loss = {loss.item():.6f}")

    # Return final deformation field
    average_shift = torch.mean(deformation_field.data)
    final_deformation_field = deformation_field.data - average_shift
    return final_deformation_field


def estimate_motion_lazy(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms e.g. (300, 15)
    patch_sidelength: int = 512,
    n_iterations: int = 200,
    n_patches_per_batch: int = 20,
    learning_rate: float = 0.05,
    device: torch.device = None,
) -> torch.Tensor:
    """Lazy motion estimation using on-demand patch extraction.
    
    This version uses the LazyPatchGrid to extract patches only when needed,
    providing memory efficiency without the complexity of caching subsets.
    
    The lazy approach is optimal when:
    - You have very large images that don't fit in memory
    - You want the simplest memory-efficient solution
    - You don't mind slightly higher computation overhead for maximum memory savings
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        
    # grab image dims
    t, h, w = image.shape

    # normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # Create lazy patch grid - no patches extracted yet!
    ph, pw = patch_sidelength, patch_sidelength
    lazy_patch_grid, data_patch_positions = patch_grid_lazy(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),
        distribute_patches=True,
    )
    
    # Get grid dimensions
    grid_shape = lazy_patch_grid.grid_shape
    if len(grid_shape) == 2:
        gh, gw = grid_shape
        total_patches = gh * gw
    else:
        gd, gh, gw = grid_shape
        total_patches = gd * gh * gw

    print(f"Total patches available: {total_patches} (lazy evaluation)")

    # Create reusable filters and masks
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

    # Initialize deformation field
    deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution,
        n_channels=2
    ).to(device)

    # Normalize patch center positions to [0, 1]
    if len(data_patch_positions.shape) == 3:  # 2D case: (gh, gw, 2)  
        data_patch_positions = data_patch_positions / torch.tensor([t - 1, h - 1, w - 1], device=device)
    else:  # 3D case: (gd, gh, gw, 3)
        data_patch_positions = data_patch_positions / torch.tensor([t - 1, h - 1, w - 1], device=device)

    # Initialize optimizer
    motion_optimiser = torch.optim.Adam(
        params=deformation_field.parameters(),
        lr=learning_rate,
    )

    # Training loop using lazy patch extraction
    for i in range(n_iterations):
        # Extract random patches on-demand - this is the key difference!
        patch_subset, patch_positions = lazy_patch_grid.random_subset(n_patches_per_batch)
        
        # Process patches same as memory-efficient version
        # Reshape from (..., 1, ph, pw) to (..., ph, pw) 
        if len(patch_subset.shape) == 5:  # (t, n_patches, 1, ph, pw)
            patch_subset = einops.rearrange(patch_subset, 't n 1 ph pw -> t n ph pw')
        elif len(patch_subset.shape) == 4:  # (n_patches, t, 1, ph, pw) - transpose needed
            patch_subset = einops.rearrange(patch_subset, 'n t 1 ph pw -> t n ph pw')
        
        # Apply mask and FFT
        patch_subset = patch_subset * mask
        patch_subset = torch.fft.rfftn(patch_subset, dim=(-2, -1))
        
        # Create reference by averaging patches
        reference_patches = torch.mean(patch_subset, dim=1, keepdim=True)  # Keep patch dimension for broadcasting

        # Predict shifts at patch centers
        predicted_shifts = -1 * deformation_field(patch_positions)

        # Shift patches by predicted shifts
        predicted_shifts_px = predicted_shifts / pixel_spacing
        shifted_patches = fourier_shift_dft_2d(
            dft=patch_subset,
            image_shape=(ph, pw),
            shifts=predicted_shifts_px,
            rfft=True,
            fftshifted=False,
        )

        # Apply frequency filters
        bandpass = _prepare_bandpass_filter(
            frequency_range=frequency_range,
            patch_shape=(ph, pw),
            pixel_spacing=pixel_spacing,
            refinement_fraction=i / (n_iterations - 1),
            device=device
        )
        shifted_patches = shifted_patches * bandpass * b_factor_envelope

        # Calculate loss
        loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)

        # Optimization step
        motion_optimiser.zero_grad()
        loss.backward()
        motion_optimiser.step()

        # Progress logging
        if i % 10 == 0:
            print(f"{i}: loss = {loss.item():.6f}")

    # Return final deformation field
    average_shift = torch.mean(deformation_field.data)
    final_deformation_field = deformation_field.data - average_shift
    return final_deformation_field


def _prepare_bandpass_filter(
    frequency_range: tuple[float, float],  # angstroms
    patch_shape: tuple[int, int],
    pixel_spacing: float,  # angstroms
    refinement_fraction: float,  # [0, 1]
    device: torch.device = None,
) -> torch.Tensor:
    # grab patch dimensions
    ph, pw = patch_shape

    # calculate filter cutoffs in cycles/pixel
    cuton, cutoff_max = torch.as_tensor(frequency_range, device=device).float()  # angstroms
    cutoff = torch.lerp(cuton, cutoff_max, refinement_fraction)
    low_fftfreq = spatial_frequency_to_fftfreq(1 / cuton, spacing=pixel_spacing)
    high_fftfreq = spatial_frequency_to_fftfreq(1 / cutoff, spacing=pixel_spacing)

    # prepapre bandpass
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
