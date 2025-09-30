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

from torch_motion_correction.patch_grid import patch_grid, patch_grid_lazy
from torch_motion_correction.utils import (
    spatial_frequency_to_fftfreq,
    normalize_image,
    prepare_bandpass_filter
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
    optimizer: str = 'adam',  # 'adam' or 'lbfgs'
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
    if optimizer.lower() == 'adam':
        motion_optimiser = torch.optim.Adam(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    elif optimizer.lower() == 'lbfgs':
        motion_optimiser = torch.optim.LBFGS(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Choose 'adam' or 'lbfgs'.")
    
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
        predicted_shifts_px = predicted_shifts
        shifted_patches = fourier_shift_dft_2d(
            dft=patch_subset,
            image_shape=(ph, pw),
            shifts=predicted_shifts_px,
            rfft=True,
            fftshifted=False,
        )  # (b, ph, pw, h, w)

        # apply fourier filters
        bandpass = prepare_bandpass_filter(
            frequency_range=frequency_range,
            patch_shape=(ph, pw),
            pixel_spacing=pixel_spacing,
            refinement_fraction=i / (n_iterations - 1),
            device=device
        )
        shifted_patches = shifted_patches * bandpass * b_factor_envelope

        # zero gradients, backpropagate loss and step optimiser
        if optimizer.lower() == 'adam':
            # calculate the loss, MSE between data patches and reference patches
            loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)
            motion_optimiser.zero_grad()
            loss.backward()
            motion_optimiser.step()
        elif optimizer.lower() == 'lbfgs':
            def closure():
                motion_optimiser.zero_grad()
                # Recompute forward pass in closure for L-BFGS
                pred_shifts = -1 * deformation_field(patch_subset_centers)
                pred_shifts_px = pred_shifts / pixel_spacing
                shift_patches = fourier_shift_dft_2d(
                    dft=patch_subset,
                    image_shape=(ph, pw),
                    shifts=pred_shifts_px,
                    rfft=True,
                    fftshifted=False,
                )
                shift_patches = shift_patches * bandpass * b_factor_envelope
                # Use same stable reference in closure
                reference_patches_closure = torch.mean(patch_subset, dim=0)
                loss = torch.mean((shift_patches - reference_patches_closure).abs() ** 2)
                loss.backward()
                return loss
            loss = motion_optimiser.step(closure)
            # Extract loss value for logging
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            else:
                loss = float(loss) if loss is not None else 0.0

        # log loss
        if i % 10 == 0:
            if isinstance(loss, torch.Tensor):
                print(f"{i}: loss = {loss.item()}")
            else:
                print(f"{i}: loss = {loss}")

    # subtract mean from deformation field data
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
    optimizer: str = 'adam',  # 'adam' or 'lbfgs'
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
    if optimizer.lower() == 'adam':
        motion_optimiser = torch.optim.Adam(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    elif optimizer.lower() == 'lbfgs':
        motion_optimiser = torch.optim.LBFGS(
            params=deformation_field.parameters(),
            lr=learning_rate,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Choose 'adam' or 'lbfgs'.")

    # Training loop using lazy patch extraction
    for i in range(n_iterations):
        # Generate random patch indices to match non-lazy behavior
        patch_subset_idx = np.random.randint(
            low=(0, 0), high=(gh, gw), size=(n_patches_per_batch, 2)
        )
        idx_gh, idx_gw = einops.rearrange(patch_subset_idx, 'b idx -> idx b')
        
        # Extract patches at the selected indices
        patch_subset, patch_positions = lazy_patch_grid.get_patches_at_indices(idx_gh, idx_gw)
        
        # Reshape from (..., 1, ph, pw) to (..., ph, pw) 
        if len(patch_subset.shape) == 5:  # (t, n_patches, 1, ph, pw)
            patch_subset = einops.rearrange(patch_subset, 't n 1 ph pw -> t n ph pw')
        elif len(patch_subset.shape) == 4:  # (n_patches, t, 1, ph, pw) - transpose needed
            patch_subset = einops.rearrange(patch_subset, 'n t 1 ph pw -> t n ph pw')
        
        # Apply mask and FFT
        patch_subset = patch_subset * mask
        patch_subset = torch.fft.rfftn(patch_subset, dim=(-2, -1))
        
        # Use the middle frame as the reference patch
        #middle_frame = patch_subset.shape[0] // 2
        reference_patches = torch.mean(patch_subset, dim=0)

        # Get patch centers for the selected patches (matching non-lazy version)
        patch_subset_centers = data_patch_positions[:, idx_gh, idx_gw]

                # Apply frequency filters can be applied outside if not keep fraction
        bandpass = prepare_bandpass_filter(
            frequency_range=frequency_range,
            patch_shape=(ph, pw),
            pixel_spacing=pixel_spacing,
            #refinement_fraction=i / (n_iterations - 1),
            device=device
        )

        patch_subset = patch_subset * bandpass * b_factor_envelope
        
        # Predict shifts at patch centers
        predicted_shifts = -1 * deformation_field(patch_subset_centers)
        
        # Shift patches by predicted shifts
        predicted_shifts_px = predicted_shifts
        shifted_patches = fourier_shift_dft_2d(
            dft=patch_subset,
            image_shape=(ph, pw),
            shifts=predicted_shifts_px,
            rfft=True,
            fftshifted=False,
        )


        #shifted_patches = shifted_patches * bandpass * b_factor_envelope

        # Optimization step
        if optimizer.lower() == 'adam':
            # Calculate loss
            loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)
            motion_optimiser.zero_grad()
            loss.backward()
            motion_optimiser.step()
        elif optimizer.lower() == 'lbfgs':
            def closure():
                motion_optimiser.zero_grad()
                # Recompute forward pass in closure for L-BFGS
                pred_shifts = -1 * deformation_field(patch_subset_centers)
                pred_shifts_px = pred_shifts
                shift_patches = fourier_shift_dft_2d(
                    dft=patch_subset,
                    image_shape=(ph, pw),
                    shifts=pred_shifts_px,
                    rfft=True,
                    fftshifted=False,
                )
                shift_patches = shift_patches * bandpass * b_factor_envelope
                # Use same stable reference in closure
                reference_patches_closure = torch.mean(patch_subset, dim=0)
                loss = torch.mean((shift_patches - reference_patches_closure).abs() ** 2)
                loss.backward()
                return loss
            loss = motion_optimiser.step(closure)
            # Extract loss value for logging
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            else:
                loss = float(loss) if loss is not None else 0.0

        # Progress logging
        if i % 10 == 0:
            if isinstance(loss, torch.Tensor):
                print(f"{i}: loss = {loss.item():.6f}")
            else:
                print(f"{i}: loss = {loss:.6f}")

    # Return final deformation field
    average_shift = torch.mean(deformation_field.data)
    final_deformation_field = deformation_field.data - average_shift
    return final_deformation_field
