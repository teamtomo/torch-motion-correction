import torch
import torch.nn.functional as F
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d

from torch_motion_correction.evaluate_deformation_grid import evaluate_deformation_grid_lazy


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
    lazy_evaluator = evaluate_deformation_grid_lazy(
        deformation_grid=deformation_grid,
        coordinates=tyx,
    )
    # Extract the actual tensor values from the lazy evaluator
    shifts_angstroms = lazy_evaluator.evaluate_all()
    shifts_px = shifts_angstroms / pixel_spacing

    # find pixel positions to sample image data at, accounting for deformations
    deformed_pixel_coords = pixel_grid + shifts_px

    # sample original image data
    corrected_frame = sample_image_2d(
        image=frame,
        coordinates=deformed_pixel_coords,
        interpolation='bicubic',
    )

    return corrected_frame
