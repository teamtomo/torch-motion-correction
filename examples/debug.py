import mrcfile
import torch

from torch_motion_correction import correct_motion, estimate_motion_cross_correlation_whole_image, \
    estimate_local_motion, correct_motion_slow

IMAGE_FILE = "../EMPIAR-10164-image-0deg-with-simulated-deformations.mrc"
PIXEL_SPACING = 0.675 * 8
LEARNED_DEFORMATION_FIELD_RESOLUTION = (20, 5, 5)  # (t, h, w)
PATCH_SIDELENGTH = 128
N_ITERATIONS = 100
N_PATCHES_PER_BATCH = 200
LEARNING_RATE = 0.05

# load data
multi_frame_micrograph = torch.as_tensor(mrcfile.read(IMAGE_FILE)).float()

# simulate whole frame
# frame = multi_frame_micrograph[0]
# shifts = torch.randint(low=-5, high=5, size=(10, 2))
# from torch_fourier_shift.fourier_shift_image import fourier_shift_image_2d
# multi_frame_micrograph = fourier_shift_image_2d(
#     image=frame, shifts=shifts
# )


# estimate whole frame motion
# deformation_field = estimate_motion_cross_correlation_whole_image(
#     image=multi_frame_micrograph,
#     pixel_spacing=PIXEL_SPACING,
#     frequency_range=(300, 10),
# )

# estimate local motion
deformation_field = estimate_local_motion(
    image=multi_frame_micrograph,
    pixel_spacing=PIXEL_SPACING,
    initial_deformation_field=None,
    deformation_field_resolution=LEARNED_DEFORMATION_FIELD_RESOLUTION,
    frequency_range=(300, 10),
    patch_sidelength=PATCH_SIDELENGTH,
    n_patches_per_batch=N_PATCHES_PER_BATCH,
    learning_rate=LEARNING_RATE,
    n_iterations=N_ITERATIONS
)


motion_corrected = correct_motion(
    image=multi_frame_micrograph,
    deformation_field=deformation_field
)




import napari

viewer = napari.Viewer()
viewer.add_image(multi_frame_micrograph.detach())
viewer.add_image(motion_corrected.detach())
napari.run()
