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

from torch_motion_correction.patch_grid import (
    patch_grid,
    patch_grid_lazy,
    patch_grid_centers,
)
from torch_motion_correction.evaluate_deformation_field import resample_deformation_field
from torch_motion_correction.utils import (
    normalize_image,
    prepare_bandpass_filter,
)


class ImagePatchIterator:
    """Helper data class for iterating over image patched around defined control points.

    NOTE: Patches will be extracted on the same device as the image.

    Attributes
    ----------
        image: The input image to be patched (t, H, W).
        image_shape: Shape of the image to be patched (t, H, W).
        patch_size: Size of the patches to extract (ph, pw).
        control_points: Control points in pixel coordinates (t, gh, gw, 3).
        control_points_normalized: Control points normalized to [0, 1] (t, gh, gw, 3).

    Methods
    -------
        get_iterator(batch_size: int) -> Iterator[torch.Tensor, torch.Tensor]:
    """

    image: torch.Tensor  # (t, H, W)
    image_shape: tuple[int, int, int]  # (t, H, W)
    patch_size: tuple[int, int]  # (ph, pw)
    control_points: torch.Tensor  # (t, gh, gw, 3)
    control_points_normalized: torch.Tensor  # (t, gh, gw, 3)

    _points_constant_over_time: bool

    def __init__(
        self,
        image: torch.Tensor,
        image_shape: tuple[int, int, int],
        patch_size: tuple[int, int],
        control_points: torch.Tensor,
    ) -> None:
        """Initialization from image shape, patch size, and control points.

        NOTE: Control points are expected to be in (t, gh, gw, 3) format, and only
        constant control points over time are currently supported.
        """
        self.image = image
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.control_points = control_points.to(image.device)

        # Normalize control points to [0, 1] in all dimensions
        t, H, W = image_shape
        self.control_points_normalized = control_points.clone().float()
        self.control_points_normalized[..., 0] /= float(t - 1)
        self.control_points_normalized[..., 1] /= float(H - 1)
        self.control_points_normalized[..., 2] /= float(W - 1)

        # Check if all time slices (zeroth dimension) have the same control point
        # positions in x-y space
        self._points_constant_over_time = torch.all(
            control_points[0, :, :, 1:] == control_points[:, :, :, 1:]
        ).item()

        if not self._points_constant_over_time:
            raise NotImplementedError(
                "Control points varying over time not supported yet"
            )

    def get_iterator(
        self, batch_size: int = 1
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Returns an iterator over image patches and normalized control points.

        Each iteration will yield a stack of image patches with shape
        (batch_size, t, ph, pw) where (ph, pw) is the patch size, and a stack of
        normalized control points with shape (batch_size, t, 3) where 3 corresponds
        to (time, y, x) coordinates in normalized [0, 1] space.

        Parameters
        ----------
            batch_size: Number of patches to return simultaneously. Default is 1.

        Returns
        -------
            Iterator yielding tuples of (image_patches, control_points).
        """

        def inner_iterator():
            """Helper function implementing the iterator logic."""
            t, gh, gw, _ = self.control_points.shape
            ph, pw = self.patch_size

            # NOTE: This is currently assuming control points are constant over time
            _control_points = self.control_points[0].reshape(-1, 3)  # (gh * gw, 3)
            _control_points_normalized = self.control_points_normalized[0].reshape(
                -1, 3
            )

            for i in range(0, gh * gw, batch_size):
                batch_control_points = _control_points[i : i + batch_size]  # (b, 3)
                batch_control_points_normalized = _control_points_normalized[
                    i : i + batch_size
                ]

                # Use actual control points to extract patches from the image
                patches = []
                for cp in batch_control_points:
                    _, y, x = cp.long()  # NOTE: this will floor float coords...

                    # NOTE: This is assuming no clipping on the boundaries
                    start_y = y - ph // 2
                    end_y = start_y + ph
                    start_x = x - pw // 2
                    end_x = start_x + pw

                    patch = self.image[:, start_y:end_y, start_x:end_x]  # (t, ph, pw)
                    patches.append(patch)

                patches = torch.stack(patches)  # (b, t, ph, pw)

                yield patches, batch_control_points_normalized  # (b, t, 3)

        return inner_iterator()


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
    optimizer: str = "adam",  # 'adam' or 'lbfgs'
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
    data_patches = einops.rearrange(data_patches, "t gh gw 1 ph pw -> t gh gw ph pw")
    gh, gw = data_patch_positions.shape[1:3]

    # apply a soft circular mask on data patches in real space
    mask = circle(
        radius=pw / 4, image_shape=(ph, pw), smoothing_radius=pw / 8, device=device
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
        device=device,
    )

    # initialise the deformation field with learnable parameters for shifts
    # grid data: (2, nt, nh, nw)
    deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution, n_channels=2
    ).to(device)

    # normalize patch center positions to [0, 1] for evaluation of shifts.
    data_patch_positions = data_patch_positions / torch.tensor(
        [t - 1, h - 1, w - 1], device=device
    )

    # initialise optimiser and detach data
    if optimizer.lower() == "adam":
        motion_optimiser = torch.optim.Adam(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    elif optimizer.lower() == "lbfgs":
        motion_optimiser = torch.optim.LBFGS(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Choose 'adam' or 'lbfgs'."
        )

    data_patches = data_patches.detach()

    # optimise shifts at grid points on deformation field
    for i in range(n_iterations):
        # take a random subset of the patch grid over spatial dimensions
        patch_subset_idx = np.random.randint(
            low=(0, 0), high=(gh, gw), size=(n_patches_per_batch, 2)
        )
        idx_gh, idx_gw = einops.rearrange(patch_subset_idx, "b idx -> idx b")
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
            device=device,
        )
        shifted_patches = shifted_patches * bandpass * b_factor_envelope

        # zero gradients, backpropagate loss and step optimiser
        if optimizer.lower() == "adam":
            # calculate the loss, MSE between data patches and reference patches
            loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)
            motion_optimiser.zero_grad()
            loss.backward()
            motion_optimiser.step()
        elif optimizer.lower() == "lbfgs":

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
                loss = torch.mean(
                    (shift_patches - reference_patches_closure).abs() ** 2
                )
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


def estimate_motion_new(
    image: torch.Tensor,  # (t, H, W)
    pixel_spacing: float,  # Angstroms
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    patch_size: tuple[int, int],  # (ph, pw)
    device: torch.device = None,
    n_iterations: int = 100,
) -> torch.Tensor:
    """Estimate motion (new method)

    TODO: Docstring
    """
    device = device if device is not None else image.device
    image = image.to(device)
    t, h, w = image.shape

    # Normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # Create the patch grid
    patch_positions = patch_grid_centers(
        image_shape=(t, h, w),
        patch_shape=(1, *patch_size),
        patch_step=(1, patch_size[0] // 2, patch_size[1] // 2),  # Default 50%
        distribute_patches=True,
        device=device,
    )  # (t, gh, gw, 3)
    print("patch_positions.shape", patch_positions.shape)
    gh, gw = patch_positions.shape[1:3]


    # TODO: Include other image pre-processing setups

    image_patch_iterator = ImagePatchIterator(
        image=image,
        image_shape=(t, h, w),
        patch_size=patch_size,
        control_points=patch_positions,
    )

    # Initialize deformation field
    deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution, n_channels=2
    )
    deformation_field.to(device)

    # TODO: Optimizer setup

    for iter_idx in range(n_iterations):

        ### DEBUGGING: Print iteration info
        print("iter_idx", iter_idx)
        
        patch_iter = image_patch_iterator.get_iterator(batch_size=8)
        
        for (patch_subset, patch_subset_centers) in patch_iter:
            # patch_subset: (b, t, ph, pw)
            # positions_subset: (b, t, 3)

            patch_subset = patch_subset * mask  # TODO re-define mask
            patch_subset = torch.fft.rfftn(patch_subset, dim=(-2, -1))
            
            # Use mean of all patches (for each batch)
            reference_patches = torch.mean(patch_subset, dim=1)  # (b, ph, pw)
            print("reference_patches.shape", reference_patches.shape)
            
            # Predict the shifts at patch centers
            predicted_shifts = -1 * deformation_field(patch_subset_centers)
            print("predicted_shifts.shape", predicted_shifts.shape)
            
            # Shift the patches by the predicted shifts
            predicted_shifts_px = predicted_shifts  # NOTE: check for angstrom scaling
            shifted_patches = fourier_shift_dft_2d(
                dft=patch_subset,
                image_shape=patch_size,
                shifts=predicted_shifts_px,
                rfft=True,
                fftshifted=False,
            )
            print("shifted_patches.shape", shifted_patches.shape)
            
            # TODO: bandpass filter and B-factor envelope
            
            # TODO: optimizer step
            
    # Return final deformation field
    # QUESTION: Why are these commented out?
    # average_shift = torch.mean(deformation_field.data)
    # final_deformation_field = deformation_field.data - average_shift
    final_deformation_field = deformation_field.data - average_shift
    return final_deformation_field


def estimate_motion_lazy(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor,  # (yx, nt, nh, nw)
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms e.g. (300, 15)
    patch_sidelength: int = 512,
    n_iterations: int = 200,
    n_patches_per_batch: int = 20,
    learning_rate: float = 0.05,
    optimizer: str = "adam",  # 'adam' or 'lbfgs'
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
    # grab image and deformation field dims
    t, h, w = image.shape
    nt, nh, nw = deformation_field_resolution

    # initialize deformation field
    # semantics: resample existing to target resolution or initialize as all zeros
    if initial_deformation_field is None:
        deformation_field = CubicCatmullRomGrid3d(
            resolution=deformation_field_resolution,
            n_channels=2
        ).to(device)
    else:
        deformation_field_data = resample_deformation_field(
            deformation_field=initial_deformation_field,
            target_resolution=(nt, nh, nw),
        )
        deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field_data).to(device)

    # normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # create lazy patch grid - no patches extracted yet!
    ph, pw = patch_sidelength, patch_sidelength
    lazy_patch_grid, data_patch_positions = patch_grid_lazy(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),
        distribute_patches=True,
    )

    # Get grid dimensions
    # get patch grid dimensions
    grid_shape = lazy_patch_grid.grid_shape
    if len(grid_shape) == 2:
        gh, gw = grid_shape
        total_patches = gh * gw
    else:
        gd, gh, gw = grid_shape
        total_patches = gd * gh * gw

    print(f"Total patches available: {total_patches} (lazy evaluation)")

    # create reusable filters and masks
    mask = circle(
        radius=pw / 4, image_shape=(ph, pw), smoothing_radius=pw / 8, device=device
    )

    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=(ph, pw),
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device,
    )

    # Normalize patch center positions to [0, 1]
    if len(data_patch_positions.shape) == 3:  # 2D case: (gh, gw, 2)
        data_patch_positions = data_patch_positions / torch.tensor(
            [t - 1, h - 1, w - 1], device=device
        )
    else:  # 3D case: (gd, gh, gw, 3)
        data_patch_positions = data_patch_positions / torch.tensor(
            [t - 1, h - 1, w - 1], device=device
        )

    # Initialize optimizer
    if optimizer.lower() == "adam":
        motion_optimiser = torch.optim.Adam(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    elif optimizer.lower() == "lbfgs":
        motion_optimiser = torch.optim.LBFGS(
            params=deformation_field.parameters(),
            lr=learning_rate,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Choose 'adam' or 'lbfgs'."
        )

    # Training loop using lazy patch extraction
    for i in range(n_iterations):
        # Generate random patch indices to match non-lazy behavior
        patch_subset_idx = np.random.randint(
            low=(0, 0), high=(gh, gw), size=(n_patches_per_batch, 2)
        )
        idx_gh, idx_gw = einops.rearrange(patch_subset_idx, "b idx -> idx b")

        # Extract patches at the selected indices
        patch_subset, patch_positions = lazy_patch_grid.get_patches_at_indices(
            idx_gh, idx_gw
        )

        # Reshape from (..., 1, ph, pw) to (..., ph, pw)
        if len(patch_subset.shape) == 5:  # (t, n_patches, 1, ph, pw)
            patch_subset = einops.rearrange(patch_subset, "t n 1 ph pw -> t n ph pw")
        elif (
            len(patch_subset.shape) == 4
        ):  # (n_patches, t, 1, ph, pw) - transpose needed
            patch_subset = einops.rearrange(patch_subset, "n t 1 ph pw -> t n ph pw")

        # Apply mask and FFT
        patch_subset = patch_subset * mask
        patch_subset = torch.fft.rfftn(patch_subset, dim=(-2, -1))

        # Use the middle frame as the reference patch
        # middle_frame = patch_subset.shape[0] // 2
        reference_patches = torch.mean(patch_subset, dim=0)

        # Get patch centers for the selected patches (matching non-lazy version)
        patch_subset_centers = data_patch_positions[:, idx_gh, idx_gw]

        # Apply frequency filters can be applied outside if not keep fraction
        bandpass = prepare_bandpass_filter(
            frequency_range=frequency_range,
            patch_shape=(ph, pw),
            pixel_spacing=pixel_spacing,
            # refinement_fraction=i / (n_iterations - 1),
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

        # shifted_patches = shifted_patches * bandpass * b_factor_envelope

        # Optimization step
        if optimizer.lower() == "adam":
            # Calculate loss
            loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)
            motion_optimiser.zero_grad()
            loss.backward()
            motion_optimiser.step()
        elif optimizer.lower() == "lbfgs":

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
                loss = torch.mean(
                    (shift_patches - reference_patches_closure).abs() ** 2
                )
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
