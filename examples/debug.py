import mrcfile
import torch

from torch_motion_correction import estimate_local_motion, correct_motion

IMAGE_FILE = "../EMPIAR-10164-image-0deg-with-simulated-deformations.mrc"
PIXEL_SPACING = 0.675 * 8
LEARNED_DEFORMATION_FIELD_RESOLUTION = (20, 5, 5)  # (t, h, w)
PATCH_SIDELENGTH = 128
N_ITERATIONS = 1000
N_PATCHES_PER_BATCH = 40
LEARNING_RATE = 0.05

# load data
multi_frame_micrograph = torch.as_tensor(mrcfile.read(IMAGE_FILE)).float()

# estimate motion
deformation_grid = estimate_local_motion(
    image=multi_frame_micrograph,
    pixel_spacing=PIXEL_SPACING,
    deformation_field_resolution=LEARNED_DEFORMATION_FIELD_RESOLUTION,
    frequency_range=(300, 10),
    patch_sidelength=PATCH_SIDELENGTH,
    learning_rate=LEARNING_RATE,
    n_iterations=N_ITERATIONS
)

motion_corrected = correct_motion(
    image=multi_frame_micrograph,
    pixel_spacing=PIXEL_SPACING,
    deformation_grid=deformation_grid
)


import napari

viewer = napari.Viewer()
viewer.add_image(multi_frame_micrograph.detach())
viewer.add_image(motion_corrected.detach())
napari.run()
