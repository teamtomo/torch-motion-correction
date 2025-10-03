import eerfile
import mrcfile
import numpy as np
import torch
from torch_motion_correction import (
    estimate_local_motion,
    correct_motion,
    correct_motion_batched,
    correct_motion_fast,
    estimate_local_motion,
    estimate_global_motion,
    estimate_motion_cross_correlation_patches,
    write_deformation_field_to_csv,
    read_deformation_field_from_csv,
)
     
from torch_fourier_filter.dose_weight import dose_weight_movie
from tifffile import TiffFile
import os

#params at top

#eer reader
eer_file = "xenon_131_000_0.0.eer"
total_fluence = 54.75
dose_per_frame = 1.0

#gain correct
gain_file = "20240627_210834_EER_GainReference.gain"
flip_gain = 0
rot_gain = 0

#motion estimate and correct
pixel_size = 0.936 # Angstroms

#dose weight
pre_exposure = 0.0 # seconds
voltage = 300.0 # kV

def load_gain_reference(gain_file: str) -> np.ndarray:
    """
    Load gain reference from either .mrc or .gain (TIFF) file.
    
    Args:
        gain_file: Path to gain reference file (.mrc or .gain)
        
    Returns:
        2D numpy array containing the gain reference
    """
    if gain_file.lower().endswith('.gain'):
        print(f"Loading gain reference from TIFF file: {gain_file}")
        # .gain files are TIFF format
        with TiffFile(gain_file) as tif:
            gain_map = tif.asarray().astype(np.float32)
    elif gain_file.lower().endswith('.mrc'):
        print(f"Loading gain reference from MRC file: {gain_file}")
        with mrcfile.open(gain_file) as f:
            gain_map = f.data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported gain file format: {gain_file}. Only .mrc and .gain files are supported.")
    
    return gain_map

def create_gain_reference_files(gain_map: np.ndarray, output_dir: str = ".") -> tuple[str, str]:
    """
    Create multiplicative and divisor gain reference MRC files.
    
    Args:
        gain_map: 2D numpy array containing the gain reference
        output_dir: Directory to save the output files
        
    Returns:
        Tuple of (multiplicative_file_path, divisor_file_path)
    """
    multiplicative_path = os.path.join(output_dir, "gain_multiplicative.mrc")
    divisor_path = os.path.join(output_dir, "gain_divisor.mrc")
    
    print(f"Creating multiplicative gain reference: {multiplicative_path}")
    with mrcfile.new(multiplicative_path, overwrite=True) as f:
        f.set_data(gain_map.astype(np.float32))
    
    print(f"Creating divisor gain reference: {divisor_path}")
    # Create inverse for divisor (handle division by zero)
    gain_divisor = np.where(gain_map != 0, 1.0 / gain_map, 1.0)
    with mrcfile.new(divisor_path, overwrite=True) as f:
        f.set_data(gain_divisor.astype(np.float32))
    
    return multiplicative_path, divisor_path

def gain_correct(
        movie : np.ndarray, 
        gain_file : str, 
        flip_gain : int, 
        rot_gain : int
) -> np.ndarray:
    """
    Apply gain correction to movie frames.
    
    Args:
        movie: Movie array with shape (n_frames, height, width)
        gain_file: Path to gain reference file (.mrc or .gain)
        flip_gain: Flip gain map (0=no flip, 1=flipY, 2=flipX)
        rot_gain: Rotate gain map (number of 90-degree rotations)
        
    Returns:
        Gain-corrected movie array
    """
    # Load gain map (handles both .mrc and .gain files)
    gain_map = load_gain_reference(gain_file)
    
    # Create gain reference files if input is .gain
    if gain_file.lower().endswith('.gain'):
        multiplicative_path, divisor_path = create_gain_reference_files(gain_map, "ttMotionCor")
        print(f"Created gain reference files: {multiplicative_path}, {divisor_path}")
    
    # Apply transformations to gain map
    if flip_gain == 1:
        gain_map = np.flip(gain_map, axis=0)  # flipY
    elif flip_gain == 2:
        gain_map = np.flip(gain_map, axis=1)  # flipX
    
    if rot_gain != 0:
        gain_map = np.rot90(gain_map, k=-rot_gain)

    return movie * gain_map

def remove_hot_pixels(movie: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    Remove hot pixels from movie frames by replacing pixels that are more than 
    threshold standard deviations above OR below the mean with a random adjacent pixel value.
    
    Args:
        movie: Movie array with shape (n_frames, height, width)
        threshold: Number of standard deviations above/below mean to consider as hot pixel
        
    Returns:
        Movie array with hot pixels replaced
    """
    print(f"Removing hot pixels with threshold {threshold} standard deviations...")
    
    movie_corrected = movie.copy()
    n_frames, height, width = movie.shape
    
    for frame_idx in range(n_frames):
        frame = movie_corrected[frame_idx]
        
        # Calculate mean and std for this frame
        frame_mean = np.mean(frame)
        frame_std = np.std(frame)
        
        # Find hot pixels (pixels above OR below threshold * std from mean)
        hot_pixel_mask = (frame > (frame_mean + threshold * frame_std)) | (frame < (frame_mean - threshold * frame_std))
        hot_pixel_coords = np.where(hot_pixel_mask)
        
        if len(hot_pixel_coords[0]) > 0:
            print(f"  Frame {frame_idx}: Found {len(hot_pixel_coords[0])} hot pixels")
            
            # Replace each hot pixel with a random adjacent pixel
            for y, x in zip(hot_pixel_coords[0], hot_pixel_coords[1]):
                # Define the 8-connected neighborhood bounds
                y_min = max(0, y - 1)
                y_max = min(height - 1, y + 1)
                x_min = max(0, x - 1)
                x_max = min(width - 1, x + 1)
                
                # Get adjacent pixels (excluding the hot pixel itself)
                adjacent_pixels = []
                for adj_y in range(y_min, y_max + 1):
                    for adj_x in range(x_min, x_max + 1):
                        if adj_y != y or adj_x != x:  # Exclude the hot pixel itself
                            adjacent_pixels.append(frame[adj_y, adj_x])
                
                # Replace with random adjacent pixel value
                if adjacent_pixels:
                    replacement_value = np.random.choice(adjacent_pixels)
                    movie_corrected[frame_idx, y, x] = replacement_value
    
    return movie_corrected

def set_frames_mean_zero(movie: np.ndarray) -> np.ndarray:
    """
    Set each frame in the movie to have mean zero by subtracting the frame mean.
    Uses vectorized operations for improved performance.
    
    Args:
        movie: Movie array with shape (n_frames, height, width)
        
    Returns:
        Movie array with each frame having mean zero
    """
    print(f"Setting each frame to mean zero (vectorized)...")
    
    # Calculate mean for each frame along the spatial dimensions (axis=(1,2))
    frame_means = np.mean(movie, axis=(1, 2), keepdims=True)
    
    # Subtract the mean from each frame using broadcasting
    movie_mean_zero = movie - frame_means
    
    n_frames = movie.shape[0]
    print(f"  Completed mean zero correction for {n_frames} frames")
    
    return movie_mean_zero

def motion_estimate_and_correct(movie : torch.Tensor, pixel_size : float) -> torch.Tensor:
    print(f"Estimating motion...")
    print(f"Testing on gpu")
    movie = movie.to(device="cuda:1")
    
    deformation_grid = estimate_local_motion(
        image=movie,
        pixel_spacing=pixel_size,
        deformation_field_resolution=(54, 6, 6),  
        patch_sidelength=1024,
        n_patches_per_batch=10,  
        learning_rate=1,      
        n_iterations=5000,        
        optimizer='lbfgs',       # Try L-BFGS
        b_factor=500,
        frequency_range=(300, 10),
    ) 
    
    '''
    deformation_grid = estimate_motion(
        image=movie,
        pixel_spacing=pixel_size,
        deformation_field_resolution=(54, 5, 5),  # (nt, nh, nw)
        frequency_range=(300, 10),  # angstroms
        patch_sidelength=2048,
        n_patches_per_batch=1,
        learning_rate=0.05,
        n_iterations=1000,
        optimizer="adam",
    ) 
    '''
    print(f"Correcting motion...")
    motion_corrected = correct_motion(
        image=movie,
        deformation_grid=deformation_grid
    )
    return motion_corrected

def motion_estimate_and_correct_cross_correlation(movie : torch.Tensor, pixel_size : float) -> torch.Tensor:
    print(f"Estimating motion cross correlation...")
    movie = movie.to(device="cuda:1")

    shifts = estimate_global_motion(
        image=movie,
        pixel_spacing=pixel_size,
        reference_frame=None,  
        b_factor=500,
        frequency_range=(300, 10)
    )

    #corrected = correct_motion_whole_image(movie, shifts, fill_mode='noise')
    
    corrected = correct_motion_fast(
        image=movie,
        deformation_grid=shifts
    )
    
    return corrected


def motion_estimate_and_correct_cross_correlation_refined(movie : torch.Tensor, pixel_size : float) -> torch.Tensor:
    print(f"Estimating motion cross correlation patches...")
    movie = movie.to(device="cuda:1")
    refined_images, deformation_grid = refine_alignment(
        images=movie,
        pixel_spacing=pixel_size,
        max_iterations=20,
        b_factor=500.0,
        mask_central_cross=True,
        max_shift_convergence_threshold=0.001,
        number_of_frames_for_running_average=1,
        savitzky_golay_window_size=7
    )
    '''
    corrected = correct_motion(
        image=movie,
        pixel_spacing=pixel_size,
        deformation_grid=deformation_grid
    )
    '''

    return refined_images

def motion_estimate_and_correct_cross_correlation_patches(movie : torch.Tensor, pixel_size : float) -> torch.Tensor:
    print(f"Estimating motion cross correlation patches...")
    movie = movie.to(device="cuda:1")
    deformation_grid = estimate_global_motion(
        image=movie,
        pixel_spacing=pixel_size,
        reference_frame=None,  
        b_factor=500,
        frequency_range=(300, 10)
    )


    num_iterations = 5
    previous_deformation_grid = deformation_grid
    for i in range(num_iterations):
        print(f"Previous deformation grid shape: {previous_deformation_grid.shape}")
        deformation_grid, deformation_field_patches = estimate_motion_cross_correlation_patches(
            image=movie,
            pixel_spacing=pixel_size,
            patch_sidelength=1024,
            deformation_field=previous_deformation_grid
        )
        print(f"Deformation grid shape: {deformation_grid.shape}")
        if i > 0:
            difference = torch.abs(deformation_grid - previous_deformation_grid)
            print(f"Difference between deformation grids: {torch.mean(difference)}")
            #if torch.mean(difference) < 0.001:
                #print(f"Deformation grids converged after {i} iterations")
                #break
        previous_deformation_grid = deformation_grid

    write_deformation_field_to_csv(deformation_grid, deformation_field_patches, "deformation_grid.csv", 54)
    #shifts = refine_correlation_patches(
    #    image=movie,
    #    pixel_spacing=pixel_size,
    #    patch_sidelength=512,
    #)
    
    corrected = correct_motion(
        image=movie,
        deformation_grid=deformation_grid
    )
    return corrected

def dose_weight(movie : torch.Tensor) -> torch.Tensor:
    # get the height and width from the last two dimensions
    frame_shape = (movie.shape[-2], movie.shape[-1])
    #FFT  each frame
    movie_dft = torch.fft.rfft2(movie, dim=(-2, -1), norm='ortho')
    #apply dose weight
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
    )
    #inverse FFT
    movie_dw = torch.fft.irfft2(movie_dw_dft, s=frame_shape, dim=(-2, -1), norm='ortho')
    image_dw = torch.sum(movie_dw, dim=0)
    return image_dw


def main():
    #render movie
    print(f"Rendering movie {eer_file}...")
    movie = eerfile.render(eer_file, dose_per_output_frame=dose_per_frame, total_fluence=total_fluence)
    # Save non-aligned mrc movie here
    #print(f"Saving non-aligned mrc movie {eer_file}...")
    #with mrcfile.new(f"motion_corr_test/non_aligned_movie_{eer_file[:-4]}.mrc", overwrite=True) as f:
    #    f.set_data(movie.astype(np.float32))
    #gain correct
    print(f"Gain correcting movie {eer_file}...")
    movie = gain_correct(movie, gain_file, flip_gain, rot_gain)
    # Save gain-corrected movie as MRC
    #print(f"Saving gain-corrected movie to ttMotionCor/movie_gain_corrected.mrc ...")
    
    #with mrcfile.new("ttMotionCor/movie_gain_corrected.mrc", overwrite=True) as f:
    #    f.set_data(movie.astype(np.float32))
    
    #remove hot pixels
    print(f"Removing hot pixels from movie {eer_file}...")
    movie = remove_hot_pixels(movie, threshold=10.0)
    #set each frame to mean zero
    print(f"Setting frames to mean zero for movie {eer_file}...")
    movie = set_frames_mean_zero(movie)
    
    # Calculate non-aligned sum BEFORE motion correction
    print(f"Calculating non-aligned sum {eer_file}...")
    non_aligned_sum = np.sum(movie, axis=0)

    
    #motion estimate and correct

    print(f"Motion estimating and correcting movie {eer_file}...")
    movie = torch.as_tensor(movie).to(torch.float32)
    #move to gpu
    movie = movie.to(device="cuda:1")

    movie_whole_image = motion_estimate_and_correct_cross_correlation(movie, pixel_size)

    #movie_patches = motion_estimate_and_correct_cross_correlation_patches(movie, pixel_size)
    #movie_patches = motion_estimate_and_correct_cross_correlation_refined(movie, pixel_size)

    movie = motion_estimate_and_correct(movie_whole_image, pixel_size)

    print(f"summing movie {eer_file}...")
    non_dw_sum = torch.sum(movie, dim=0)
    non_dw_sum_whole_image = torch.sum(movie_whole_image, dim=0)
    #non_dw_sum_patches = torch.sum(movie_patches, dim=0)
    #non_dw_sum_refine_deformation_field = torch.sum(movie_refine_deformation_field, dim=0)
    print(f"Dose weighting movie {eer_file}...")
    dw_sum = dose_weight(movie)
    dw_sum_whole_image = dose_weight(movie_whole_image)
    #dw_sum_patches = dose_weight(movie_patches)
    #dw_sum_refine_deformation_field = dose_weight(movie_refine_deformation_field)
    # Move tensors to cpu if they are on gpu before saving
    non_dw_sum_cpu = non_dw_sum.cpu() if non_dw_sum.is_cuda else non_dw_sum
    dw_sum_cpu = dw_sum.cpu() if dw_sum.is_cuda else dw_sum
    non_dw_sum_whole_image_cpu = non_dw_sum_whole_image.cpu() if non_dw_sum_whole_image.is_cuda else non_dw_sum_whole_image
    dw_sum_whole_image_cpu = dw_sum_whole_image.cpu() if dw_sum_whole_image.is_cuda else dw_sum_whole_image
    #non_dw_sum_patches_cpu = non_dw_sum_patches.cpu() if non_dw_sum_patches.is_cuda else non_dw_sum_patches
    #dw_sum_patches_cpu = dw_sum_patches.cpu() if dw_sum_patches.is_cuda else dw_sum_patches
    #non_dw_sum_refine_deformation_field_cpu = non_dw_sum_refine_deformation_field.cpu() if non_dw_sum_refine_deformation_field.is_cuda else non_dw_sum_refine_deformation_field
    #dw_sum_refine_deformation_field_cpu = dw_sum_refine_deformation_field.cpu() if dw_sum_refine_deformation_field.is_cuda else dw_sum_refine_deformation_field
    # Convert to numpy for processing and saving
    non_dw_sum_numpy = non_dw_sum_cpu.numpy()
    dw_sum_numpy = dw_sum_cpu.numpy()
    non_dw_sum_whole_image_numpy = non_dw_sum_whole_image_cpu.numpy()
    dw_sum_whole_image_numpy = dw_sum_whole_image_cpu.numpy()
    #non_dw_sum_patches_numpy = non_dw_sum_patches_cpu.numpy()
    #dw_sum_patches_numpy = dw_sum_patches_cpu.numpy()
    #non_dw_sum_refine_deformation_field_numpy = non_dw_sum_refine_deformation_field_cpu.numpy()
    #dw_sum_refine_deformation_field_numpy = dw_sum_refine_deformation_field_cpu.numpy()
    # Save original versions (without hot pixel removal on final images)
    print(f"Saving original sums {eer_file}...")
    with mrcfile.new(f"ttMotionCor/non_dw_sum_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        f.set_data(non_dw_sum_numpy)
    with mrcfile.new(f"ttMotionCor/dw_sum_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        f.set_data(dw_sum_numpy)
    with mrcfile.new(f"ttMotionCor/non_dw_sum_whole_image_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        f.set_data(non_dw_sum_whole_image_numpy)
    with mrcfile.new(f"ttMotionCor/dw_sum_whole_image_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        f.set_data(dw_sum_whole_image_numpy)
    #with mrcfile.new(f"ttMotionCor/non_dw_sum_patches_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        #f.set_data(non_dw_sum_patches_numpy)
    #with mrcfile.new(f"ttMotionCor/dw_sum_patches_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        #f.set_data(dw_sum_patches_numpy)
    #with mrcfile.new(f"ttMotionCor/non_dw_sum_refine_deformation_field_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        #f.set_data(non_dw_sum_refine_deformation_field_numpy)
    #with mrcfile.new(f"ttMotionCor/dw_sum_refine_deformation_field_original2_{eer_file[:-4]}.mrc", overwrite=True) as f:
        #f.set_data(dw_sum_refine_deformation_field_numpy)
    '''
    # Remove hot pixels from final images
    print(f"Removing hot pixels from final non-dose-weighted sum...")
    non_dw_sum_hp_corrected = remove_hot_pixels_2d_mean(non_dw_sum_numpy, threshold=10.0)
    
    print(f"Removing hot pixels from final dose-weighted sum...")
    dw_sum_hp_corrected = remove_hot_pixels_2d_mean(dw_sum_numpy, threshold=10.0)
    
    # Save hot pixel corrected versions
    print(f"Saving hot pixel corrected sums {eer_file}...")
    with mrcfile.new(f"ttMotionCor/non_dw_sum_{eer_file[:-4]}.mrc", overwrite=True) as f:
        f.set_data(non_dw_sum_hp_corrected)
    with mrcfile.new(f"ttMotionCor/dw_sum_{eer_file[:-4]}.mrc", overwrite=True) as f:
        f.set_data(dw_sum_hp_corrected)
    '''
    # Save non-aligned sum (calculated before motion correction)
    #print(f"Saving non-aligned sum {eer_file}...")
    #with mrcfile.new(f"ttMotionCor/non_aligned_sum_{eer_file[:-4]}.mrc", overwrite=True) as f:
        #f.set_data(np.float32(non_aligned_sum))


    


if __name__ == "__main__":
    main()




