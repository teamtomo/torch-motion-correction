import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple


def write_deformation_field_to_csv(
    deformation_field: torch.Tensor,
    data_patch_positions: torch.Tensor,
    output_path: Union[str, Path],
    num_real_frames: int,
    device: torch.device = None
) -> None:
    """
    Write deformation field to CSV format.
    
    Parameters
    ----------
    deformation_field: torch.Tensor
        Deformation field with shape (2, t, gh, gw)
        Channel 0 is y_shift, channel 1 is x_shift
    data_patch_positions: torch.Tensor
        Patch center positions with shape (t, gh, gw, 3)
        Last dimension contains (t_index, center_y, center_x) coordinates
    output_path: Union[str, Path]
        Path to output CSV file
    num_real_frames: int
        Number of real movie frames for frame index calculation
    device: torch.device, optional
        Device for computation
        
    Returns
    -------
    None
    """
    if device is None:
        device = deformation_field.device
    else:
        deformation_field = deformation_field.to(device)
        data_patch_positions = data_patch_positions.to(device)
    
    # Get dimensions
    _, t, gh, gw = deformation_field.shape
    
    # Prepare data lists
    frame_indices = []
    center_y_coords = []
    center_x_coords = []
    y_shifts = []
    x_shifts = []
    
    # Calculate frame mapping
    frame_step = num_real_frames / t
    
    # Iterate through all patches and time points
    for t_idx in range(t):
        # Calculate real frame index
        real_frame_idx = frame_step * t_idx
        
        for gy in range(gh):
            for gx in range(gw):
                # Get patch center coordinates (skip t_index at position 0)
                center_y = data_patch_positions[t_idx, gy, gx, 1].item()
                center_x = data_patch_positions[t_idx, gy, gx, 2].item()
                
                # Get shifts
                y_shift = deformation_field[0, t_idx, gy, gx].item()
                x_shift = deformation_field[1, t_idx, gy, gx].item()
                
                # Append to lists
                frame_indices.append(real_frame_idx)
                center_y_coords.append(center_y)
                center_x_coords.append(center_x)
                y_shifts.append(y_shift)
                x_shifts.append(x_shift)
    
    # Create DataFrame
    df = pd.DataFrame({
        'frame': frame_indices,
        'center_y': center_y_coords,
        'center_x': center_x_coords,
        'y_shift': y_shifts,
        'x_shift': x_shifts
    })
    
    # Write to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Deformation field written to {output_path}")
    print(f"CSV contains {len(df)} rows with {t} time points and {gh*gw} patches per time point")


def read_deformation_field_from_csv(
    csv_path: Union[str, Path],
    num_real_frames: int,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read deformation field from CSV format.
    
    Parameters
    ----------
    csv_path: Union[str, Path]
        Path to input CSV file
    num_real_frames: int
        Number of real movie frames for frame index calculation
    device: torch.device, optional
        Device for output tensors
        
    Returns
    -------
    deformation_field: torch.Tensor
        Deformation field with shape (2, t, gh, gw)
    data_patch_positions: torch.Tensor
        Patch center positions with shape (t, gh, gw, 3)
    """
    if device is None:
        device = torch.device('cpu')
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print(f"Reading deformation field from {csv_path}")
    print(f"CSV contains {len(df)} rows")
    
    # Get unique frame indices and sort them
    unique_frames = sorted(df['frame'].unique())
    t = len(unique_frames)
    
    # Calculate frame step for validation
    expected_frame_step = num_real_frames / t
    
    # Get unique patch positions for the first frame to determine grid size
    first_frame_data = df[df['frame'] == unique_frames[0]]
    unique_centers_y = sorted(first_frame_data['center_y'].unique())
    unique_centers_x = sorted(first_frame_data['center_x'].unique())
    
    gh = len(unique_centers_y)
    gw = len(unique_centers_x)
    
    print(f"Detected grid dimensions: t={t}, gh={gh}, gw={gw}")
    print(f"Expected frame step: {expected_frame_step:.3f}")
    
    # Initialize tensors
    deformation_field = torch.zeros((2, t, gh, gw), device=device, dtype=torch.float32)
    data_patch_positions = torch.zeros((t, gh, gw, 3), device=device, dtype=torch.float32)
    
    # Create mapping from center coordinates to grid indices
    center_y_to_gy = {center_y: gy for gy, center_y in enumerate(unique_centers_y)}
    center_x_to_gx = {center_x: gx for gx, center_x in enumerate(unique_centers_x)}
    
    # Fill tensors
    for t_idx, frame_idx in enumerate(unique_frames):
        frame_data = df[df['frame'] == frame_idx]
        
        for _, row in frame_data.iterrows():
            center_y = row['center_y']
            center_x = row['center_x']
            y_shift = row['y_shift']
            x_shift = row['x_shift']
            
            # Find grid indices
            gy = center_y_to_gy[center_y]
            gx = center_x_to_gx[center_x]
            
            # Fill deformation field
            deformation_field[0, t_idx, gy, gx] = y_shift  # Channel 0 = y_shift
            deformation_field[1, t_idx, gy, gx] = x_shift  # Channel 1 = x_shift
            
            # Fill patch positions (t_index, center_y, center_x)
            data_patch_positions[t_idx, gy, gx, 0] = t_idx  # t_index
            data_patch_positions[t_idx, gy, gx, 1] = center_y  # center_y
            data_patch_positions[t_idx, gy, gx, 2] = center_x  # center_x
    
    print(f"Successfully loaded deformation field with shape {deformation_field.shape}")
    print(f"Deformation field range: y=[{deformation_field[0].min():.3f}, {deformation_field[0].max():.3f}], "
          f"x=[{deformation_field[1].min():.3f}, {deformation_field[1].max():.3f}]")
    
    return deformation_field, data_patch_positions


def validate_deformation_field_csv(csv_path: Union[str, Path]) -> dict:
    """
    Validate the structure and content of a deformation field CSV file.
    
    Parameters
    ----------
    csv_path: Union[str, Path]
        Path to CSV file to validate
        
    Returns
    -------
    validation_info: dict
        Dictionary containing validation results and statistics
    """
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_columns = ['frame', 'center_y', 'center_x', 'y_shift', 'x_shift']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get statistics
    unique_frames = sorted(df['frame'].unique())
    t = len(unique_frames)
    
    # Check grid consistency
    first_frame_data = df[df['frame'] == unique_frames[0]]
    unique_centers_y = sorted(first_frame_data['center_y'].unique())
    unique_centers_x = sorted(first_frame_data['center_x'].unique())
    
    gh = len(unique_centers_y)
    gw = len(unique_centers_x)
    
    expected_rows_per_frame = gh * gw
    actual_rows_per_frame = len(first_frame_data)
    
    # Validate all frames have same number of patches
    frame_counts = df.groupby('frame').size()
    inconsistent_frames = frame_counts[frame_counts != expected_rows_per_frame]
    
    validation_info = {
        'total_rows': len(df),
        'num_frames': t,
        'grid_height': gh,
        'grid_width': gw,
        'expected_rows_per_frame': expected_rows_per_frame,
        'actual_rows_per_frame': actual_rows_per_frame,
        'frame_range': (unique_frames[0], unique_frames[-1]),
        'y_shift_range': (df['y_shift'].min(), df['y_shift'].max()),
        'x_shift_range': (df['x_shift'].min(), df['x_shift'].max()),
        'inconsistent_frames': len(inconsistent_frames),
        'is_valid': len(missing_columns) == 0 and len(inconsistent_frames) == 0
    }
    
    return validation_info
