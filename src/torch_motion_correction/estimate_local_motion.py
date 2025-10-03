import tqdm
import einops
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_filter.envelopes import b_envelope
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import circle

from torch_motion_correction.optimization_state import OptimizationTracker
from torch_motion_correction.patch_grid import (
    patch_grid_centers,
)
from torch_motion_correction.deformation_field_utils import (
    resample_deformation_field,
)
from torch_motion_correction.patch_utils import ImagePatchIterator
from torch_motion_correction.utils import (
    normalize_image,
    prepare_bandpass_filter,
)


def estimate_local_motion(
    image: torch.Tensor,  # (t, H, W)
    pixel_spacing: float,  # Angstroms
    patch_shape: tuple[int, int],  # (ph, pw)
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    device: torch.device = None,
    n_iterations: int = 100,
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms
    optimizer_type: str = "adam",
    optimizer_kwargs: dict | None = None,
    return_trajectory: bool = False,
    trajectory_kwargs: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, OptimizationTracker]:
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
    torch.Tensor | tuple[torch.Tensor, OptimizationTracker]
        The estimated deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts. If `return_trajectory` is True, also returns an
        OptimizationTrajectory object containing the optimization history.
    """
    device = device if device is not None else image.device
    image = image.to(device)
    t, h, w = image.shape
    ph, pw = patch_shape

    if return_trajectory:
        trajectory_kwargs = trajectory_kwargs if trajectory_kwargs is not None else {}
        trajectory = OptimizationTracker(**trajectory_kwargs)

    # Normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # Create the patch grid
    patch_positions = patch_grid_centers(
        image_shape=(t, h, w),
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),  # Default 50% overlap
        distribute_patches=True,
        device=device,
    )  # (t, gh, gw, 3)

    gh, gw = patch_positions.shape[1:3]

    print("Making new deformation field")
    new_deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution, n_channels=2
    ).to(device)
    print("New deformation field made")
    print("new_deformation_field.shape", new_deformation_field.data.shape)

    if initial_deformation_field is None:
        deformation_field_data = torch.zeros(size=(2, *deformation_field_resolution), device=device)
    elif initial_deformation_field is not None:
        deformation_field_data = resample_deformation_field(
            deformation_field=initial_deformation_field.detach(),
            target_resolution=(
                deformation_field_resolution[0], deformation_field_resolution[1], deformation_field_resolution[2]),
        )
        deformation_field_data -= torch.mean(deformation_field_data)
        # deformation_field_data *= -1
        print(f"Resampled initial deformation field to {deformation_field_data.shape}")
    print("Making deformation field")
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field_data).to(device)
    print("Deformation field made")
    print("deformation_field.shape", deformation_field.data.shape)

    # Reusable masks and Fourier filters
    # NOTE: This is assuming square patches... revisit if needed
    circle_mask = circle(
        radius=patch_shape[1] / 4,
        image_shape=patch_shape,
        smoothing_radius=patch_shape[1] / 4,
        device=device,
    )

    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=patch_shape,
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device,
    )

    bandpass_filter = prepare_bandpass_filter(
        frequency_range=frequency_range,
        patch_shape=patch_shape,
        pixel_spacing=pixel_spacing,
        refinement_fraction=1.0,  # Not used in this context
        device=device,
    )

    # Instantiate the patch iterator (mini-batch like data-loader)
    image_patch_iterator = ImagePatchIterator(
        image=image,
        patch_size=patch_shape,
        control_points=patch_positions,
    )

    motion_optimizer = _setup_optimizer(
        optimizer_type=optimizer_type,
        parameters=new_deformation_field.parameters(),
        **(optimizer_kwargs if optimizer_kwargs is not None else {}),
    )

    # "Training" loop going over all patched n_iterations times
    pbar = tqdm.tqdm(range(n_iterations))
    for iter_idx in pbar:
        patch_iter = image_patch_iterator.get_iterator(batch_size=8)  # TODO: expose
        total_loss = 0.0
        n_batches = 0

        for patch_batch, patch_batch_centers in patch_iter:
            # patch_subset: (b, t, ph, pw)
            # positions_subset: (b, t, 3)

            patch_batch = patch_batch * circle_mask
            patch_batch = torch.fft.rfftn(patch_batch, dim=(-2, -1))

            # # Use mean of all patches (for each batch)
            # reference_patches = torch.mean(patch_subset, dim=1, keepdim=True)
            # # print("reference_patches.shape", reference_patches.shape)

            # Predict the shifts based on the deformation field and apply those
            # shifts to the patches. Shifted patches are use to compute loss relative
            # to the mean of the patches in the batch.
            shifted_patches, predicted_shifts_angstroms = _compute_shifted_patches_and_shifts(
                initial_deformation_field=deformation_field,
                new_deformation_field=new_deformation_field,
                patch_batch=patch_batch,
                patch_batch_centers=patch_batch_centers,
                pixel_spacing=pixel_spacing,
                ph=ph,
                pw=pw,
                b_factor_envelope=b_factor_envelope,
                bandpass=bandpass_filter,
            )

            # Calculate mean of all shifted patches over time to use as a reference
            reference_patches = torch.mean(shifted_patches, dim=1, keepdim=True)

            # Calculate loss, normalized by number of pixels in patches
            loss = torch.mean((shifted_patches - reference_patches).abs() ** 2) / (ph * pw)

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

            # Use gradient accumulation to optimize over all patches simultaneously
            loss.backward()
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            total_loss += loss_value
            n_batches += 1

        # Step the optimizer after each pass over whole image
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
    final_deformation_field = new_deformation_field.data + deformation_field.data
    average_shift = torch.mean(final_deformation_field.data)
    final_deformation_field = final_deformation_field.data - average_shift

    if return_trajectory:
        return final_deformation_field, trajectory
    else:
        return final_deformation_field


def _compute_shifted_patches_and_shifts(
    initial_deformation_field: CubicCatmullRomGrid3d,
    new_deformation_field: CubicCatmullRomGrid3d,
    patch_batch: torch.Tensor,
    patch_batch_centers: torch.Tensor,
    pixel_spacing: float,
    ph: int,
    pw: int,
    b_factor_envelope: torch.Tensor = None,
    bandpass: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the forward pass for motion estimation for a batch of patches.

    Parameters
    ----------
    initial_deformation_field : CubicCatmullRomGrid3d
        The deformation field model to predict shifts.
    new_deformation_field : CubicCatmullRomGrid3d
        The new deformation field model to predict shifts.
    patch_batch : torch.Tensor
        A batch of image patches in Fourier space with shape (b, t, ph, pw).
    patch_batch_centers : torch.Tensor
        Normalized control point centers for the batch with shape (b, t, 3).
    pixel_spacing : float
        Pixel spacing in Angstroms.
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
    predicted_shifts = -1 * (
        new_deformation_field(patch_batch_centers)
        + initial_deformation_field(patch_batch_centers)
    )
    predicted_shifts = einops.rearrange(predicted_shifts, "b t yx -> t b yx")
    predicted_shifts_px = predicted_shifts / pixel_spacing


    # Shift the patches by the predicted shifts
    shifted_patches = fourier_shift_dft_2d(
        dft=patch_batch,
        image_shape=(ph, pw),
        shifts=predicted_shifts_px,
        rfft=True,
        fftshifted=False,
    )  # (b, t, ph, pw//2 + 1)

    # Apply Fourier filters
    if bandpass is not None:
        shifted_patches = shifted_patches * bandpass

    if b_factor_envelope is not None:
        shifted_patches = shifted_patches * b_factor_envelope

    return shifted_patches, predicted_shifts


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


