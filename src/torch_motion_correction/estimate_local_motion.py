import tqdm
import einops
import torch
import mrcfile
import contextlib
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

from torch_motion_correction.data_io import write_deformation_field_to_csv

from torch_motion_correction.correct_motion import correct_motion, correct_motion_two_grids

from torch_fourier_filter.dose_weight import dose_weight_movie

#from leopard_em.pydantic_models.managers import InspectPeaksManager

from leopard_em.backend.core_inspect_peaks import core_inspect_peaks

from torch_motion_correction.gradient_tracer import (
    trace_tensor_gradient_info, 
    check_pipeline_gradients,
    add_gradient_debugging_to_pipeline
)

from torch_motion_correction.fft_debug_hooks import (
    FFTDebugger,
    find_cpu_tensors_in_computation_graph,
    add_device_check_hooks
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
        deformation_field_data -= torch.mean(deformation_field_data, dim=(1, 2, 3), keepdim=True)
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

            # Calculate weighted mean of shifted patches over time to use as a reference
            # Linear weights: earlier frames have higher weights
            n_patches = shifted_patches.shape[1]
            weights = torch.linspace(1.0, 0.0, n_patches, device=shifted_patches.device, dtype=shifted_patches.dtype)
            weights = einops.rearrange(weights, 'n -> 1 n 1 1')  # Reshape for broadcasting
            
            # Normalize weights to sum to 1
            weights = weights / weights.sum(dim=1, keepdim=True)
            
            reference_patches = torch.sum(shifted_patches * weights, dim=1, keepdim=True)

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
                deformation_field=new_deformation_field.data,
                loss=average_loss,
                step=iter_idx,
            )

    # Return final deformation field
    final_deformation_field = new_deformation_field.data + deformation_field.data
    average_shift = torch.mean(final_deformation_field.data, dim=(1, 2, 3), keepdim=True)
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



def estimate_local_motion_2dtm(
    image: torch.Tensor,  # (t, H, W)
    pixel_spacing: float,  # Angstroms
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    refine_config_path: str,
    pre_exposure: float = 0.0,
    dose_per_frame: float = 1.0,
    voltage: float = 300.0,
    device: torch.device = None,
    n_iterations: int = 100,
    optimizer_type: str = "adam",
    optimizer_kwargs: dict | None = None,
    return_trajectory: bool = False,
    trajectory_kwargs: dict | None = None,
    debug_gradients: bool = False,  # Enable detailed gradient debugging
) -> torch.Tensor | tuple[torch.Tensor, OptimizationTracker]:
    """Estimate motion (new method)

    Parameters
    ----------
    image: torch.Tensor
        (t, H, W) image to estimate motion from where t is the number of frames,
        H is the height, and W is the width.
    pixel_spacing: float
        Pixel spacing in Angstroms.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw) where nt is the number of
        time points, nh is the number of control points in height, and nw is the
        number of control points in width.
    initial_deformation_field: torch.Tensor | None
        Initial deformation field to start from with shape (2, nt, nh, nw) where 2
        corresponds to (y, x) shifts. If None, initializes to zero shifts.
    refine_config_path: str
        Path to the refine config file.
    pre_exposure: float
        Pre-exposure time in seconds. Default is 0.0.
    dose_per_frame: float
        Dose per frame in electrons per pixel. Default is 1.0.
    voltage: float
        Voltage in kV. Default is 300.0.
    device: torch.device, optional
        Device to perform computation on. If None, uses the device of the input image.
    n_iterations: int
        Number of iterations for the optimization process. Default is 100.
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

    if return_trajectory:
        trajectory_kwargs = trajectory_kwargs if trajectory_kwargs is not None else {}
        trajectory = OptimizationTracker(**trajectory_kwargs)

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
            deformation_field=initial_deformation_field,
            target_resolution=(
                deformation_field_resolution[0], deformation_field_resolution[1], deformation_field_resolution[2]),
        )
        deformation_field_data -= torch.mean(deformation_field_data, dim=(1, 2, 3), keepdim=True)
        # deformation_field_data *= -1
        print(f"Resampled initial deformation field to {deformation_field_data.shape}")
    print("Making deformation field")
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field_data).to(device)
    print("Deformation field made")
    print("deformation_field.shape", deformation_field.data.shape)


    motion_optimizer = torch.optim.Adam(
        params=new_deformation_field.parameters(),
        lr=2e-1,
        betas = (0.9, 0.999),
        eps = 1e-08,
        weight_decay = 0,
        amsgrad = False,

    )



    # Add device check hooks to parameters (only if debugging)
    if debug_gradients:
        print("\n🔍 Adding device check hooks to parameters")
        add_device_check_hooks(new_deformation_field.named_parameters())

    # "Training" loop going over all patched n_iterations times
    pbar = tqdm.tqdm(range(n_iterations))
    all_losses = []
    
    # Conditionally wrap with FFT debugger
    fft_debugger_context = FFTDebugger() if debug_gradients else contextlib.nullcontext()
    
    with fft_debugger_context as fft_debug:
        for iter_idx in pbar:
            # Let the optimizer learn freely - no mean-zero constraint during optimization
            # Mean-zero will be applied only at the final result
            
            write_deformation_field_to_csv(new_deformation_field.data, f"def_fields/new_deformation_field_{iter_idx}.csv")
            write_deformation_field_to_csv(new_deformation_field.data + deformation_field.data, f"def_fields/total_deformation_field_{iter_idx}.csv")
            torch.cuda.empty_cache()
            #new_deformation_field.data = torch.zeros_like(new_deformation_field.data) # reset to zero so it should work every time

            #current_deformation_field = (new_deformation_field.data + deformation_field.data).to(device)
            # Get the parameter directly (preserves gradients)
            # Use two-grid approach - pass grid objects directly!
            # This preserves gradients through new_deformation_field
            if debug_gradients:
                print("\n🔍 Step 1: Using two-grid approach")
                print(f"new_deformation_field has parameters: {list(new_deformation_field.named_parameters())}")
                print(f"deformation_field (base) will be frozen")

            print("Correcting motion with two grids")
            print(f"  new_deformation_grid type: {type(new_deformation_field)}")
            print(f"  new_deformation_grid has parameters: {any(p.requires_grad for p in new_deformation_field.parameters())}")
            print(f"  base_deformation_grid type: {type(deformation_field)}")
            
            # No mean-zero constraint during optimization - let the optimizer learn freely

            corrected_movie = correct_motion_two_grids(
                image=image.clone(),
                new_deformation_grid=new_deformation_field,  # Pass grid object with gradients!
                base_deformation_grid=deformation_field,      # Pass base grid object (frozen)
                pixel_spacing=pixel_spacing,
                grad=True,
                device=device
            )
            '''

            print(f"  ✅ correct_motion_two_grids completed")
            
            print(f"2. After correct_motion:")
            print(f"   corrected_movie.requires_grad: {corrected_movie.requires_grad}")
            print(f"   corrected_movie.grad_fn: {corrected_movie.grad_fn}")
            if not corrected_movie.requires_grad:
                print("   ❌ GRADIENT LOST at correct_motion!")
            
            if debug_gradients:
                print("\n🔍 Step 2: Checking corrected movie")
                trace_tensor_gradient_info(corrected_movie, "corrected_movie")

            '''
            #corrected_movie = correct_motion(image, new_deformation_field.data + deformation_field.data, pixel_spacing, grad=True)

            '''
            print(f"\nChecking if corrected_movie depends on new_deformation_field:")
            print(f"  Tracing backward graph:")

            # Trace corrected_movie's graph to see if it includes grid operations
            if corrected_movie.grad_fn is not None:
                found_grid_ops = False
                current_fn = corrected_movie.grad_fn
                grid_related_ops = ['AddBackward', 'CatBackward', 'IndexBackward', 
                                   'SliceBackward', 'MulBackward', 'GridSamplerBackward']
                found_ops = []
                
                for i in range(30):  # Trace more steps to find grid ops
                    if current_fn is None:
                        break
                    fn_name = str(current_fn)
                    print(f"\n    Step {i}: {fn_name}")
                    
                    # Check if this is a grid-related operation
                    for op in grid_related_ops:
                        if op in fn_name:
                            found_ops.append(op)
                            if op == 'AddBackward':
                                # This is where new_shifts + base_shifts happens
                                print(f"      ✅ Found {op} - deformation field IS in graph!")
                                found_grid_ops = True
                                break
                    
                    if found_grid_ops:
                        break
                        
                    if hasattr(current_fn, 'next_functions') and current_fn.next_functions:
                        current_fn = current_fn.next_functions[0][0]
                    else:
                        break
                
                print(f"\n  Found operations: {found_ops}")
                if found_grid_ops:
                    print("  ✅ Gradient path to deformation field FOUND!")
                else:
                    print("  ⚠️  AddBackward not found in first 30 steps")
                    print("  This is expected with grid interpolation - checking actual gradient flow...")

            '''
            
        # dose weight this movie
            print("Dose weighting movie")
            dw_image = _dose_weight(
            corrected_movie, pixel_spacing,
            pre_exposure=pre_exposure,
            dose_per_frame=dose_per_frame,
            voltage=voltage,
            memory_efficient=True,
            chunk_size=1,
            memory_strategy='full',
            )

            with mrcfile.new(f"def_fields/dw_image_{iter_idx}_dw-checkpointing.mrc", overwrite=True) as mrc:
                mrc.set_data(dw_image.detach().cpu().numpy())

            # Compute and print the difference between this dw_image and the previous iteration's dw_image
            if iter_idx == 0:
                prev_dw_image = dw_image.detach().clone()
                print("No previous dw_image to compare (first iteration).")
            else:
                # Compute difference (L2 norm and mean absolute difference)
                diff = dw_image - prev_dw_image
                l2_diff = diff.norm().item()
                mean_abs_diff = diff.abs().mean().item()
                print(f"Difference from previous dw_image: L2 norm = {l2_diff:.6f}, mean abs = {mean_abs_diff:.6f}")
                prev_dw_image = dw_image.detach().clone()

            import time
            #time.sleep(5)

            print(f"3. After _dose_weight:")
            print(f"   dw_image.requires_grad: {dw_image.requires_grad}")
            print(f"   dw_image.grad_fn: {dw_image.grad_fn}")
            if not dw_image.requires_grad:
                print("   ❌ GRADIENT LOST at _dose_weight!")
            
            if debug_gradients:
                print("\n🔍 Step 3: Checking dose weighted image")
                trace_tensor_gradient_info(dw_image, "dw_image")
            
            if debug_gradients:
                print("\n🔍 Step 4: Calling core_inspect_peaks")
                print(f"   dw_image type: {type(dw_image)}")
                print(f"   dw_image requires_grad: {dw_image.requires_grad}")
            
            inspect_peaks_manager = InspectPeaksManager.from_yaml(refine_config_path)
            backend_kwargs = inspect_peaks_manager.make_backend_core_function_kwargs(mrc_image=dw_image)
            
            if debug_gradients:
                print(f"   backend_kwargs keys: {list(backend_kwargs.keys()) if isinstance(backend_kwargs, dict) else 'not a dict'}")
                if isinstance(backend_kwargs, dict) and 'mrc_image' in backend_kwargs:
                    mrc_img = backend_kwargs['mrc_image']
                    print(f"   backend_kwargs['mrc_image'] type: {type(mrc_img)}")
                    if isinstance(mrc_img, torch.Tensor):
                        trace_tensor_gradient_info(mrc_img, "backend_kwargs['mrc_image']")
            result = core_inspect_peaks(
                batch_size=20,
                num_cuda_streams=inspect_peaks_manager.computational_config.num_cpus,
                use_multiprocessing=False,
                **backend_kwargs,
            )
            result_dict = {
            "max_z_score": result[0],
            "max_cc": result[1],
            }
            inspect_peaks_manager.inspect_peaks_result_to_dataframe(f"def_fields/results_{iter_idx}.csv", result_dict)
            
            if debug_gradients:
                print("\n🔍 Step 5: Checking core_inspect_peaks result")
                print(f"   result type: {type(result)}")
                if isinstance(result, (tuple, list)):
                    print(f"   result length: {len(result)}")
                    for i, r in enumerate(result):
                        print(f"   result[{i}] type: {type(r)}")
                        if isinstance(r, torch.Tensor):
                            trace_tensor_gradient_info(r, f"result[{i}]")
            
            refined_mip = result[0]
            refined_scaled_mip = result[1]
            
            print(f"4. After core_inspect_peaks:")
            print(f"   refined_scaled_mip.requires_grad: {refined_scaled_mip.requires_grad}")
            print(f"   refined_scaled_mip.grad_fn: {refined_scaled_mip.grad_fn}")
            if not refined_scaled_mip.requires_grad:
                print("   ❌ GRADIENT LOST at core_inspect_peaks!")
            
            if debug_gradients:
                print("\n🔍 Step 6: Checking tensors before loss computation")
                trace_tensor_gradient_info(refined_mip, "refined_mip")
                trace_tensor_gradient_info(refined_scaled_mip, "refined_scaled_mip")


            
            loss = -torch.mean(refined_scaled_mip)
            all_losses.append(-1*loss.item())
            
            print(f"5. Final loss:")
            print(f"   loss.requires_grad: {loss.requires_grad}")
            print(f"   loss.grad_fn: {loss.grad_fn}")
            if not loss.requires_grad:
                print("   ❌ GRADIENT LOST at loss computation!")
            print("="*70 + "\n")
            
            if debug_gradients:
                print("\n🔍 Step 7: Checking loss")
                trace_tensor_gradient_info(loss, "loss")
            
            if debug_gradients:
                print("\n🔍 Step 8: Full pipeline gradient check")
                add_gradient_debugging_to_pipeline(
                    deformation_field=new_deformation_field.data,
                    corrected_movie=corrected_movie,
                    dw_image=dw_image,
                    result=result,
                    loss=loss,
                    parameters=new_deformation_field.named_parameters()
                )
            
            if debug_gradients:
                print("\n🔍 Checking for CPU tensors in computation graph before backward:")
                cpu_tensors = find_cpu_tensors_in_computation_graph(loss)
                if cpu_tensors:
                    print(f"   ❌ FOUND {len(cpu_tensors)} CPU TENSORS - This will cause MKL FFT errors!")
                    for depth, tensor in cpu_tensors[:5]:  # Show first 5
                        print(f"      Depth {depth}: shape={tensor.shape}, dtype={tensor.dtype}")
                else:
                    print(f"   ✅ No CPU tensors found")
            # Try backward
            if debug_gradients:
                print("\n🔍 Step 9: Attempting backward pass")

            # Right before loss.backward() (around line 658):
            print("\n" + "="*70)
            print("CHECKING BACKWARD GRAPH CONNECTION")
            print("="*70)

            # Check if loss is actually connected to the parameters
            print("Parameters that loss depends on:")
            params_in_graph = set()
            for name, param in new_deformation_field.named_parameters():
                if param.grad_fn is not None or param.requires_grad:
                    # Try to see if this parameter is in the computational graph
                    print(f"  {name}: requires_grad={param.requires_grad}, is_leaf={param.is_leaf}")
        
            # Check loss backward graph
            print(f"\nLoss backward graph:")
            print(f"  loss.grad_fn: {loss.grad_fn}")

            if loss.grad_fn is not None:
                print(f"  loss.grad_fn.next_functions: {loss.grad_fn.next_functions}")
    
                # Trace back a few steps
                current_fn = loss.grad_fn
                for i in range(5):  # Trace 5 steps back
                    if current_fn is None:
                        break
                    print(f"  Step {i}: {current_fn}")
                    if hasattr(current_fn, 'next_functions') and current_fn.next_functions:
                        next_fn = current_fn.next_functions[0][0]
                        current_fn = next_fn
                    else:
                        print(f"    (no more functions)")
                        break

            print("="*70 + "\n")
            
            motion_optimizer.zero_grad()

            try:
                loss.backward()
                # DIAGNOSTIC: Check if gradients exist and their magnitude
                print(f"\n{'='*60}")
                print(f"Iteration {iter_idx} - Gradient Check")
                print(f"{'='*60}")

                grad_exists = False
                for name, param in new_deformation_field.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_mean = param.grad.mean().item()
                        grad_max = param.grad.abs().max().item()
                        param_norm = param.data.norm().item()
        
                        print(f"{name}:")
                        print(f"  Grad norm: {grad_norm:.10f}")
                        print(f"  Grad mean: {grad_mean:.10f}")
                        print(f"  Grad max:  {grad_max:.10f}")
                        print(f"  Param norm: {param_norm:.10f}")
                        
                        if grad_norm > 1e-10:
                            grad_exists = True
                    else:
                        print(f"{name}: NO GRADIENT!")

                if not grad_exists:
                    print("❌ NO GRADIENTS DETECTED - Loss not connected to parameters!")
                print(f"{'='*60}\n")
                # Clip very aggressively since you're destroying good alignment
                print("Clipping gradients")
                torch.nn.utils.clip_grad_norm_(
                    new_deformation_field.parameters(), 
                    max_norm=0.01  # Very small!
                )

                if debug_gradients:
                    print("✅ Backward pass succeeded!")
                    # Check if gradients were computed
                    for name, param in new_deformation_field.named_parameters():
                        if param.grad is not None:
                            print(f"✅ {name} has gradients: mean={param.grad.mean().item():.6f}")
                        else:
                            print(f"❌ {name} has NO gradients!")
                        
            except RuntimeError as e:
                print(f"❌ Backward pass FAILED with error:")
                print(f"   {str(e)}")
                if debug_gradients:
                    print("\n   This confirms gradients are broken in the pipeline.")
                    print("   Check the debug output above to find where gradients were lost.")
                raise
            
            motion_optimizer.step()

            # log loss
            if iter_idx % 1 == 0:
                print(f"{iter_idx}: mean cc = {-1*loss.item()}")

        if return_trajectory and trajectory.sample_this_step(iter_idx):
            trajectory.add_checkpoint(
                deformation_field=new_deformation_field.data,
                loss=loss,
                step=iter_idx,
            )
    # End of with FFTDebugger() context

    # Return final deformation field
    final_deformation_field = new_deformation_field.data + deformation_field.data
    average_shift = torch.mean(final_deformation_field.data, dim=(1, 2, 3), keepdim=True)
    final_deformation_field = final_deformation_field.data - average_shift

    if return_trajectory:
        return final_deformation_field, trajectory
    else:
        return final_deformation_field, all_losses


def _dose_weight(
    movie : torch.Tensor, 
    pixel_size : float, 
    pre_exposure : float = 0.0,
    dose_per_frame : float = 1.0, 
    voltage : float = 300.0,
    memory_strategy : str = 'checkpointing',
    memory_efficient : bool = True,
    chunk_size : int = 10,
    ) -> torch.Tensor:
    """
    Apply dose weighting to a movie using the correct normalization.
    
    Since dose_weight_movie requires all frames for proper normalization,
    we use memory optimization strategies instead of chunking.
    
    Parameters
    ----------
    movie : torch.Tensor
        Input movie tensor (t, h, w)
    pixel_size : float
        Pixel size in Angstroms
    pre_exposure : float
        Pre-exposure dose
    dose_per_frame : float
        Dose per frame
    voltage : float
        Acceleration voltage
    memory_strategy : str
        Memory optimization strategy: 'full', 'checkpointing', 'adaptive'
    """
    frame_shape = (movie.shape[-2], movie.shape[-1])
    
    if memory_strategy == 'full':
        # Direct computation with chunked FFT and inverse FFT
        n_frames = movie.shape[0]
        movie_dft_chunks = []
        
        # Process forward FFT in chunks
        for i in range(0, n_frames, chunk_size):
            chunk = movie[i:min(i+chunk_size, n_frames)]
            chunk_dft = torch.fft.rfft2(chunk, dim=(-2, -1), norm='ortho')
            movie_dft_chunks.append(chunk_dft)
            del chunk, chunk_dft
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        movie_dft = torch.cat(movie_dft_chunks, dim=0)
        del movie_dft_chunks
        torch.cuda.empty_cache()
        
        movie_dw_dft = dose_weight_movie(
            movie_dft=movie_dft,
            image_shape=frame_shape,
            pixel_size=pixel_size,
            pre_exposure=pre_exposure,
            dose_per_frame=dose_per_frame,
            voltage=voltage,
            crit_exposure_bfactor=-1,
            rfft=True,
            fftshift=False,
            memory_efficient=memory_efficient,
            chunk_size=chunk_size,
        )
        
        # Process inverse FFT in chunks to reduce memory
        n_frames = movie_dw_dft.shape[0]
        image_dw = None
        
        for i in range(0, n_frames, chunk_size):
            chunk_dw_dft = movie_dw_dft[i:min(i+chunk_size, n_frames)]
            chunk_dw = torch.fft.irfft2(chunk_dw_dft, s=frame_shape, dim=(-2, -1), norm='ortho')
            
            if image_dw is None:
                image_dw = torch.sum(chunk_dw, dim=0)
            else:
                image_dw += torch.sum(chunk_dw, dim=0)
            
            del chunk_dw, chunk_dw_dft
            torch.cuda.empty_cache()
        
        return image_dw
    
    elif memory_strategy == 'checkpointing':
        # Use gradient checkpointing to reduce memory usage
        def _dose_weight_forward(movie):
            n_frames = movie.shape[0]
            movie_dft_chunks = []
            
            # Process forward FFT in chunks
            for i in range(0, n_frames, chunk_size):
                chunk = movie[i:min(i+chunk_size, n_frames)]
                chunk_dft = torch.fft.rfft2(chunk, dim=(-2, -1), norm='ortho')
                movie_dft_chunks.append(chunk_dft)
                del chunk, chunk_dft
                torch.cuda.empty_cache()
            
            # Concatenate all chunks
            movie_dft = torch.cat(movie_dft_chunks, dim=0)
            del movie_dft_chunks
            torch.cuda.empty_cache()
            
            movie_dw_dft = dose_weight_movie(
                movie_dft=movie_dft,
                image_shape=frame_shape,
                pixel_size=pixel_size,
                pre_exposure=pre_exposure,
                dose_per_frame=dose_per_frame,
                voltage=voltage,
                crit_exposure_bfactor=-1,
                rfft=True,
                fftshift=False,
                memory_efficient=memory_efficient,
                chunk_size=chunk_size,
            )
            
            # Process inverse FFT in chunks to reduce memory
            n_frames = movie_dw_dft.shape[0]
            image_dw = None
            
            for i in range(0, n_frames, chunk_size):
                chunk_dw_dft = movie_dw_dft[i:min(i+chunk_size, n_frames)]
                chunk_dw = torch.fft.irfft2(chunk_dw_dft, s=frame_shape, dim=(-2, -1), norm='ortho')
                
                if image_dw is None:
                    image_dw = torch.sum(chunk_dw, dim=0)
                else:
                    image_dw += torch.sum(chunk_dw, dim=0)
                
                del chunk_dw, chunk_dw_dft
                torch.cuda.empty_cache()
            
            return image_dw
        
        return torch.utils.checkpoint.checkpoint(_dose_weight_forward, movie)
    else:
        raise ValueError(f"Unknown memory strategy: {memory_strategy}")
