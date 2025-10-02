from datetime import datetime
import random
from typing import Iterator

import tqdm
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
from torch_motion_correction.evaluate_deformation_field import (
    resample_deformation_field,
)
from torch_motion_correction.utils import (
    normalize_image,
    prepare_bandpass_filter,
)


class ImagePatchIterator:
    """Helper data class for iterating over image patched around defined control points.

    NOTE: Patches will be extracted on the same device as the image.

    Attributes
    ----------
    image : torch.Tensor
        The input image (movie frame stack) to draw patches from with shape
        (t, H, W) where t is the number of frames, H is height and W is width.
    image_shape : tuple[int, int, int]
        Shape of the input image (t, H, W).
    patch_size : tuple[int, int]
        Size of the patches to extract (ph, pw) in terms of pixels.
    control_points : torch.Tensor
        Control points in pixel coordinates with shape (t, gh, gw, 3) where
        gh and gw are the number of control points in height and width dimensions,
        and 3 corresponds to (time, y, x) coordinates.
    control_points_normalized : torch.Tensor
        Control points normalized to [0, 1] in all dimensions with shape

    Methods
    -------
    get_iterator(batch_size: int, randomized: bool = True) -> Iterator[tuple[torch.Tensor, torch.Tensor]]
        Data-loader style iterator yielding batches of image patches and corresponding
        normalized control points for each batch.
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
        patch_size: tuple[int, int],
        control_points: torch.Tensor,
    ) -> None:
        """Initialization from image shape, patch size, and control points.

        NOTE: Control points are expected to be in (t, gh, gw, 3) format, and only
        constant control points over time are currently supported.
        
        Parameters
        ----------
        image : torch.Tensor
            The input image to be patched (t, H, W).\
                
        """
        assert len(image.shape) == 3, "Image must be 3D (t, H, W)"
        assert len(patch_size) == 2, "Patch size must be 2D (ph, pw)"
        assert (len(control_points.shape) == 4) and (
            control_points.shape[-1] == 3
        ), "Control points must be (t, gh, gw, 3)"
        assert (
            image.shape[0] == control_points.shape[0]
        ), "Image time dimension and control points time dimension must match"

        self.image = image
        self.image_shape = image.shape
        self.patch_size = patch_size
        self.control_points = control_points.to(image.device)

        # Normalize control points to [0, 1] in all dimensions
        t, H, W = self.image_shape
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

        # Check that box extraction around extrema won't go out of bounds
        ph, pw = patch_size
        min_y = torch.min(control_points[..., 1]).item()
        max_y = torch.max(control_points[..., 1]).item()
        min_x = torch.min(control_points[..., 2]).item()
        max_x = torch.max(control_points[..., 2]).item()
        err_msg = (
            f"Patch size {patch_size} too large for control points in image "
            f"of shape {self.image_shape} where control points range from "
            f"y: [{min_y}, {max_y}], x: [{min_x}, {max_x}]"
        )
        assert min_y - ph // 2 >= 0, err_msg
        assert max_y + ph // 2 < H, err_msg
        assert min_x - pw // 2 >= 0, err_msg
        assert max_x + pw // 2 < W, err_msg

    def get_iterator(
        self, batch_size: int = 1, randomized: bool = True
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Returns an iterator over image patches and normalized control points.

        Each iteration will yield a stack of image patches with shape
        (batch_size, t, ph, pw) where (ph, pw) is the patch size, and a stack of
        normalized control points with shape (batch_size, t, 3) where 3 corresponds
        to (time, y, x) coordinates in normalized [0, 1] space.

        Parameters
        ----------
        batch_size : int
            Number of patches to return simultaneously. Default is 1.
        randomized : bool
            Whether to randomize the order of patches. Default is True.

        Returns
        -------
        Iterator[tuple[torch.Tensor, torch.Tensor]]
            An iterator yielding tuples of (patches, normalized_control_points)
            where patches is a tensor of shape (batch_size, t, ph, pw) and
            normalized_control_points is a tensor of shape (batch_size, t, 3).
        """

        def inner_iterator():
            """Helper function implementing the iterator logic."""
            t, gh, gw, _ = self.control_points.shape
            ph, pw = self.patch_size

            # NOTE: This is currently assuming control points are constant over time
            _control_points = self.control_points[0].reshape(-1, 3)  # (gh * gw, 3)
            # _control_points_norm = self.control_points_normalized[0].reshape(-1, 3)  # (t, gh * gw, 3)
            _control_points_norm = self.control_points_normalized.reshape(
                t, -1, 3
            )  # (t, gh * gw, 3)

            # Apply randomization if requested
            indices = list(range(gh * gw))
            if randomized:
                random.shuffle(indices)
                _control_points = _control_points[indices]
                _control_points_norm = _control_points_norm[:, indices]

            for i in range(0, gh * gw, batch_size):
                batch_control_points = _control_points[i : i + batch_size]  # (b, 3)
                batch_control_points_norm = _control_points_norm[
                    :, i : i + batch_size
                ]  # (b, t, 3)

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

                yield patches, batch_control_points_norm

        return inner_iterator()


class OptimizationCheckpoint:
    """Dataclass for storing single optimization stage (deformation field + loss).

    Parameters
    ----------
    deformation_field : torch.Tensor
        The deformation field at this checkpoint with shape (2, nt, nh, nw) where
        2 corresponds to (y, x) shifts.
    loss : float
        The loss value at this checkpoint.
    step : int
        The optimization step number at this checkpoint.

    Methods
    -------
    as_dict() -> dict
        Returns a dictionary representation of the checkpoint.
    """

    deformation_field: torch.Tensor  # (yx, nt, nh, nw)
    loss: float
    step: int

    def __init__(self, deformation_field: torch.Tensor, loss: float, step: int):
        self.deformation_field = deformation_field.cpu()
        self.loss = loss
        self.step = step

    def as_dict(self) -> dict:
        return {
            "deformation_field": self.deformation_field.tolist(),
            "loss": self.loss,
            "step": self.step,
        }


class OptimizationTrajectory:
    """Dataclass for storing and tracking motion correction optimization trajectory.

    Parameters
    ----------
    optimization_checkpoints : list[OptimizationCheckpoint]
        List of optimization checkpoints recorded during the optimization process.
    sample_every_n_steps : int
        Frequency of sampling checkpoints during optimization.
    total_steps : int
        Total number of optimization steps.

    Methods
    -------
    sample_this_step(step: int) -> bool
        Determines if a checkpoint should be sampled at the given step.
    add_checkpoint(deformation_field: torch.Tensor, loss: float, step: int) -> None
        Adds a new optimization checkpoint.
    as_dict() -> dict
        Returns a dictionary representation of the optimization trajectory.
    to_json(filepath: str) -> None
        Saves the optimization trajectory to a JSON file.
    """

    optimization_checkpoints: list[OptimizationCheckpoint]
    sample_every_n_steps: int
    total_steps: int

    def __init__(self, sample_every_n_steps: int, total_steps: int):
        self.optimization_checkpoints = []
        self.sample_every_n_steps = sample_every_n_steps
        self.total_steps = total_steps

    def sample_this_step(self, step: int) -> bool:
        return step % self.sample_every_n_steps == 0 or step == self.total_steps - 1

    def add_checkpoint(
        self, deformation_field: torch.Tensor, loss: float, step: int
    ) -> None:
        self.optimization_checkpoints.append(
            OptimizationCheckpoint(deformation_field, loss, step)
        )

    def as_dict(self) -> dict:
        return {
            "optimization_checkpoints": [
                cp.as_dict() for cp in self.optimization_checkpoints
            ],
            "sample_every_n_steps": self.sample_every_n_steps,
            "total_steps": self.total_steps,
        }

    def to_json(self, filepath: str) -> None:
        import json

        with open(filepath, "w") as f:
            json.dump(self.as_dict(), f)


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
        motion_optimizer = torch.optim.Adam(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    elif optimizer.lower() == "lbfgs":
        motion_optimizer = torch.optim.LBFGS(
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
            motion_optimizer.zero_grad()
            loss.backward()
            motion_optimizer.step()
        elif optimizer.lower() == "lbfgs":

            def closure():
                motion_optimizer.zero_grad()
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

            loss = motion_optimizer.step(closure)
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


def _setup_optimizer(
    optimizer_type: str,
    parameters: list[torch.Tensor],
    **kwargs,
) -> torch.optim.Optimizer:
    """Helper function to setup optimizer with given parameters and kwargs."""
    if optimizer_type.lower() == "adam":
        lr = kwargs.get("lr", 0.01)
        betas = kwargs.get("betas", (0.9, 0.999))
        eps = kwargs.get("eps", 1e-08)
        weight_decay = kwargs.get("weight_decay", 0)
        amsgrad = kwargs.get("amsgrad", False)
        return torch.optim.Adam(
            params=parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_type.lower() == "sgd":
        lr = kwargs.get("lr", 0.01)
        return torch.optim.SGD(params=parameters, lr=lr)
    elif optimizer_type.lower() == "lbfgs":
        lr = kwargs.get("lr", 1)
        max_iter = kwargs.get("max_iter", 20)
        max_eval = kwargs.get("max_eval", None)
        tolerance_grad = kwargs.get("tolerance_grad", 1e-07)
        tolerance_change = kwargs.get("tolerance_change", 1e-09)
        history_size = kwargs.get("history_size", 100)
        line_search_fn = kwargs.get("line_search_fn", "strong_wolfe")
        return torch.optim.LBFGS(
            params=parameters,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. Choose 'adam' or 'lbfgs'."
        )


def _compute_forward_pass(
    deformation_field: CubicCatmullRomGrid3d,
    patch_subset: torch.Tensor,
    batch_subset_centers: torch.Tensor,
    ph: int,
    pw: int,
    b_factor_envelope: torch.Tensor = None,
    bandpass: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the forward pass for motion estimation for a batch of patches.

    Parameters
    ----------
    deformation_field : CubicCatmullRomGrid3d
        The deformation field model to predict shifts.
    patch_subset : torch.Tensor
        A batch of image patches in Fourier space with shape (b, t, ph, pw).
    batch_subset_centers : torch.Tensor
        Normalized control point centers for the batch with shape (b, t, 3).
    ph : int
        Patch height in pixels.
    pw : int
        Patch width in pixels.
    b_factor_envelope : torch.Tensor | None
        The B-factor envelope to apply in Fourier space with shape (ph, pw//2 + 1).
        If None, no envelope is applied.
    bandpass : torch.Tensor | None
        The bandpass filter to apply in Fourier space with shape (ph, pw//2 + 1).
        If None, no bandpass is applied.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - shifted_patches: The shifted patches after applying predicted shifts and
          filters, with shape (b, t, ph, pw//2 + 1).
        - predicted_shifts: The predicted shifts from the deformation field,
          with shape (b, t, 2).
    """
    predicted_shifts = -1 * deformation_field(batch_subset_centers)
    predicted_shifts = einops.rearrange(predicted_shifts, "b t yx -> t b yx")

    # Shift the patches by the predicted shifts
    shifted_patches = fourier_shift_dft_2d(
        dft=patch_subset,
        image_shape=(ph, pw),
        shifts=predicted_shifts,
        rfft=True,
        fftshifted=False,
    )  # (b, t, ph, pw//2 + 1)

    # Apply Fourier filters
    if bandpass is not None:
        shifted_patches = shifted_patches * bandpass

    if b_factor_envelope is not None:
        shifted_patches = shifted_patches * b_factor_envelope

    return shifted_patches, predicted_shifts


# def _optimizer_step_adam(
#     motion_optimizer: torch.optim.Adam,
#     shifted_patches: torch.Tensor,
#     reference_patches: torch.Tensor,
# ) -> float:
#     """Optimizer step for Adam optimizer."""
#     # Using squared difference for loss
#     loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)
#     motion_optimizer.zero_grad()
#     loss.backward()
#     motion_optimizer.step()

#     return loss.item()


# def _optimizer_step_lbfgs(
#     motion_optimizer: torch.optim.LBFGS,
#     deformation_field: CubicCatmullRomGrid3d,
#     patch_subset: torch.Tensor,
#     patch_subset_centers: torch.Tensor,
#     patch_shape: tuple[int, int],
#     bandpass: torch.Tensor | None,
#     b_factor_envelope: torch.Tensor | None,
# ) -> float:
#     """Optimizer step for L-BFGS optimizer."""

#     def closure():
#         motion_optimizer.zero_grad()
#         # Recompute forward pass in closure for L-BFGS
#         shift_patches, _ = _compute_forward_pass(
#             deformation_field=deformation_field,
#             patch_subset=patch_subset,
#             batch_subset_centers=patch_subset_centers,
#             ph=patch_shape[0],
#             pw=patch_shape[1],
#             b_factor_envelope=b_factor_envelope,
#             bandpass=bandpass,
#         )
#         # Use same stable reference in closure
#         reference_patches_closure = torch.mean(patch_subset, dim=0)
#         loss = torch.mean((shift_patches - reference_patches_closure).abs() ** 2)
#         loss.backward()
#         return loss

#     loss = motion_optimizer.step(closure)
#     # Extract loss value for logging
#     if isinstance(loss, torch.Tensor):
#         return loss.item()
#     else:
#         return float(loss) if loss is not None else 0.0


def estimate_motion_new(
    image: torch.Tensor,  # (t, H, W)
    pixel_spacing: float,  # Angstroms
    patch_size: tuple[int, int],  # (ph, pw)
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    device: torch.device = None,
    n_iterations: int = 100,
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),
    optimizer_type: str = "adam",
    optimizer_kwargs: dict | None = None,
    return_trajectory: bool = False,
    trajectory_kwargs: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, OptimizationTrajectory]:
    """Estimate motion (new method)

    Parameters
    ----------
    image: torch.Tensor
        (t, H, W) image to estimate motion from where t is the number of frames,
        H is the height, and W is the width.
    pixel_spacing: float
        Pixel spacing in Angstroms.
    patch_size: tuple[int, int]
        Size of the patches to extract (ph, pw) in terms of pixels.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw) where nt is the number of
        time points, nh is the number of control points in height, and nw is the
        number of control points in width.
    initial_deformation_field: torch.Tensor | None
        Initial deformation field to start from with shape (2, nt, nh, nw) where 2
        corresponds to (y, x) shifts. If None, initializes to zero shifts.
    device: torch.device, optional
        Device to perform computation on. If None, uses the device of the input image.
    n_iterations: int
        Number of iterations for the optimization process. Default is 100.
    b_factor: float
        B-factor to apply in Fourier space to downweight high frequencies.
        Default is 500.
    frequency_range: tuple[float, float]
        Frequency range in Angstroms for bandpass filtering (low, high).
        Default is (300, 10).
    optimizer_type: str
        Type of optimizer to use ('adam' or 'lbfgs'). Default is 'adam'.
    optimizer_kwargs: dict | None
        Additional keyword arguments for the optimizer. If None, uses defaults.
    return_trajectory: bool
        Whether to return the optimization trajectory. Default is False. If true, a
        second return value will be provided which is an OptimizationTrajectory object.
    trajectory_kwargs: dict | None
        Additional keyword arguments for the trajectory tracking. If None, uses
        defaults.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, OptimizationTrajectory]
        The estimated deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts. If `return_trajectory` is True, also returns an
        OptimizationTrajectory object containing the optimization history.
    """
    device = device if device is not None else image.device
    image = image.to(device)
    t, h, w = image.shape

    if return_trajectory:
        trajectory_kwargs = trajectory_kwargs if trajectory_kwargs is not None else {}
        trajectory = OptimizationTrajectory(**trajectory_kwargs)

    # Normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # Initialize a deformation field using pre-existing data, if provided
    if initial_deformation_field is None:
        initial_deformation_field = torch.zeros(
            (2, *deformation_field_resolution),
            device="cpu",  # NOTE: resample_deformation_field needs to support device before making on CUDA device
        )

    deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution, n_channels=2
    ).to(device)

    # # TODO: Resupport initialization
    # deformation_field_data = resample_deformation_field(
    #     deformation_field=initial_deformation_field,
    #     target_resolution=deformation_field_resolution,
    # )
    # deformation_field_data = deformation_field_data.to(device)
    # deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field_data)
    # deformation_field = deformation_field.to(device)

    # Create the patch grid
    patch_positions = patch_grid_centers(
        image_shape=(t, h, w),
        patch_shape=(1, *patch_size),
        patch_step=(1, patch_size[0] // 2, patch_size[1] // 2),  # Default 50% overlap
        distribute_patches=True,
        device=device,
    )  # (t, gh, gw, 3)

    gh, gw = patch_positions.shape[1:3]

    # Reusable masks and Fourier filters
    # NOTE: This is assuming square patches... revisit if needed
    circle_mask = circle(
        radius=patch_size[1] / 4,
        image_shape=patch_size,
        smoothing_radius=patch_size[1] / 4,
        device=device,
    )

    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=patch_size,
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device,
    )

    # Instantiate the patch iterator (mini-batch like data-loader)
    image_patch_iterator = ImagePatchIterator(
        image=image,
        patch_size=patch_size,
        control_points=patch_positions,
    )

    motion_optimizer = _setup_optimizer(
        optimizer_type=optimizer_type,
        parameters=deformation_field.parameters(),
        **(optimizer_kwargs if optimizer_kwargs is not None else {}),
    )

    # "Training" loop going over all patched n_iterations times
    pbar = tqdm.tqdm(range(n_iterations))
    for iter_idx in pbar:
        patch_iter = image_patch_iterator.get_iterator(batch_size=8)  # TODO: expose
        total_loss = 0.0
        n_batches = 0

        for patch_subset, patch_subset_centers in patch_iter:
            # patch_subset: (b, t, ph, pw)
            # positions_subset: (b, t, 3)

            patch_subset = patch_subset * circle_mask
            patch_subset = torch.fft.rfftn(patch_subset, dim=(-2, -1))

            # Use mean of all patches (for each batch)
            reference_patches = torch.mean(patch_subset, dim=1, keepdim=True)
            # print("reference_patches.shape", reference_patches.shape)

            # Predict the shifts based on the deformation field and apply those
            # shifts to the patches. Shifted patches are use to compute loss relative
            # to the mean of the patches in the batch.
            shifted_patches, predicted_shifts = _compute_forward_pass(
                deformation_field=deformation_field,
                patch_subset=patch_subset,
                batch_subset_centers=patch_subset_centers,
                ph=patch_size[0],
                pw=patch_size[1],
                b_factor_envelope=b_factor_envelope,
                bandpass=None,  # TODO: add bandpass filter
            )

            # Use gradient accumulation to optimize over all patches simultaneously
            loss = torch.mean((shifted_patches - reference_patches).abs() ** 2)
            loss.backward()
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

            # # Optimizer step
            # if optimizer_type.lower() == "adam":
            #     loss = _optimizer_step_adam(
            #         motion_optimizer=motion_optimizer,
            #         shifted_patches=shifted_patches,
            #         reference_patches=reference_patches,
            #     )
            # elif optimizer_type.lower() == "lbfgs":
            #     loss = _optimizer_step_lbfgs(
            #         motion_optimizer=motion_optimizer,
            #         deformation_field=deformation_field,
            #         patch_subset=patch_subset,
            #         patch_subset_centers=patch_subset_centers,
            #         patch_shape=patch_size,
            #         b_factor_envelope=b_factor_envelope,
            #         bandpass=None,  # TODO: add bandpass filter
            #     )

            total_loss += loss_value
            n_batches += 1

        # Outside of patch subset loop, do the optimizer step once per iteration
        motion_optimizer.step()
        motion_optimizer.zero_grad()

        # update tqdm with current running average loss for this iteration
        current_avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        pbar.set_postfix({"avg_batch_loss": f"{current_avg_loss:.6f}"})

        # Record trajectory if requested
        average_loss = total_loss / n_batches if n_batches > 0 else 0.0
        if return_trajectory and trajectory.sample_this_step(iter_idx):
            trajectory.add_checkpoint(
                deformation_field=deformation_field.data,
                loss=average_loss,
                step=iter_idx,
            )

    # Return final deformation field
    # QUESTION: Why are these commented out?
    average_shift = torch.mean(deformation_field.data)
    # final_deformation_field = deformation_field.data - average_shift
    final_deformation_field = deformation_field.data - average_shift

    if return_trajectory:
        return final_deformation_field, trajectory
    else:
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

    if initial_deformation_field is not None:
        initial_deformation_field = initial_deformation_field.to(device)

    # grab image and deformation field dims
    t, h, w = image.shape
    nt, nh, nw = deformation_field_resolution

    # initialize deformation field
    # semantics: resample existing to target resolution or initialize as all zeros
    if initial_deformation_field is None:
        deformation_field = CubicCatmullRomGrid3d(
            resolution=deformation_field_resolution, n_channels=2
        ).to(device)
    else:
        deformation_field_data = resample_deformation_field(
            deformation_field=initial_deformation_field,
            target_resolution=(nt, nh, nw),
        )

        print(f"Resampled initial deformation field to {deformation_field_data.shape}")
        deformation_field = CubicCatmullRomGrid3d.from_grid_data(
            deformation_field_data
        ).to(device)

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
        motion_optimizer = torch.optim.Adam(
            params=deformation_field.parameters(),
            lr=learning_rate,
        )
    elif optimizer.lower() == "lbfgs":
        motion_optimizer = torch.optim.LBFGS(
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
            device=device,
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
            motion_optimizer.zero_grad()
            loss.backward()
            motion_optimizer.step()
        elif optimizer.lower() == "lbfgs":

            def closure():
                motion_optimizer.zero_grad()
                # Recompute forward pass in closure for L-BFGS
                pred_shifts = -1 * deformation_field(patch_subset_centers)
                # pred_shifts = deformation_field(patch_subset_centers)
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

            loss = motion_optimizer.step(closure)
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
