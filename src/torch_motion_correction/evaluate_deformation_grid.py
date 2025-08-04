import torch
import numpy as np
from typing import Tuple, Union, Sequence
from torch_cubic_spline_grids import CubicCatmullRomGrid3d


def evaluate_deformation_grid(
    deformation_grid: torch.Tensor,  # (nt, nh, nw)
    tyx: torch.Tensor,  # (..., 3)
) -> torch.Tensor:
    """Evaluate shifts from deformation field data."""
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_grid).to(deformation_grid.device)
    predicted_shifts = deformation_field(tyx)
    return predicted_shifts


def evaluate_deformation_grid_batched(
    deformation_grid: torch.Tensor,  # (nt, nh, nw)
    tyx: torch.Tensor,  # (..., 3)
    batch_size: int = 50000,
) -> torch.Tensor:
    """Memory-efficient evaluation by processing coordinates in batches."""
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_grid).to(deformation_grid.device)
    
    # Store original shape and flatten coordinates
    original_shape = tyx.shape
    tyx_flat = tyx.reshape(-1, 3)
    n_coords = tyx_flat.shape[0]
    
    # Process coordinates in batches
    batch_results = []
    for start_idx in range(0, n_coords, batch_size):
        end_idx = min(start_idx + batch_size, n_coords)
        batch_coords = tyx_flat[start_idx:end_idx]
        batch_shifts = deformation_field(batch_coords)
        batch_results.append(batch_shifts)
        
        # Clear cache between batches to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Combine results and restore original shape
    all_shifts = torch.cat(batch_results, dim=0)
    return all_shifts.reshape(original_shape[:-1] + (2,))


class LazyDeformationGridEvaluator:
    """
    A lazy deformation grid evaluator that computes shifts on-demand to avoid memory overhead.
    Presents the same interface as pre-computed evaluation but only evaluates coordinates
    when they are actually accessed.
    """
    
    def __init__(
        self,
        deformation_grid: torch.Tensor,  # (2, nt, nh, nw) or (nt, nh, nw)
        coordinates: torch.Tensor,  # (..., 3) - tyx coordinates
        batch_size: int = 50000,
    ):
        """
        Parameters
        ----------
        deformation_grid: torch.Tensor
            The deformation grid data. Shape (2, nt, nh, nw) or (nt, nh, nw).
        coordinates: torch.Tensor
            (..., 3) array of tyx coordinates where shifts should be evaluated.
        batch_size: int
            Batch size for processing coordinates to control memory usage.
        """
        self.deformation_grid = deformation_grid
        self.coordinates = coordinates
        self.batch_size = batch_size
        
        # Initialize the deformation field once
        self.deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_grid).to(deformation_grid.device)
        
        # Store shape information
        self.coord_shape = coordinates.shape
        self._shifts_shape = coordinates.shape[:-1] + (2,)  # Replace last dim (3) with (2)
        
        # Cache for storing computed shifts
        self._cache = {}
        self._cache_keys = set()
    
    @property
    def shape(self) -> torch.Size:
        """Return the shape that the full shifts tensor would have."""
        return torch.Size(self._shifts_shape)
    
    @property
    def device(self) -> torch.device:
        """Return the device of the deformation grid."""
        return self.deformation_grid.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the deformation grid."""
        return self.deformation_grid.dtype
    
    def __getitem__(self, key) -> torch.Tensor:
        """
        Evaluate shifts on-demand based on indexing.
        
        This is where the magic happens - only the requested coordinates are evaluated.
        """
        # Convert key to a hashable cache key
        cache_key = self._make_cache_key(key)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Extract the subset of coordinates we need
        coord_subset = self.coordinates[key]
        
        # Evaluate shifts for this subset
        if coord_subset.numel() == 0:
            # Empty selection
            shifts = torch.empty(coord_subset.shape[:-1] + (2,), 
                               device=self.device, 
                               dtype=self.dtype)
        elif coord_subset.numel() <= self.batch_size * 3:  # *3 because each coord has 3 elements
            # Small enough to process at once
            shifts = self._evaluate_subset(coord_subset)
        else:
            # Use batched processing
            shifts = self._evaluate_subset_batched(coord_subset)
        
        # Cache the result
        self._cache[cache_key] = shifts
        self._cache_keys.add(cache_key)
        
        # Limit cache size to prevent memory bloat
        if len(self._cache) > 100:  # Arbitrary limit
            self._evict_cache()
        
        return shifts
    
    def _make_cache_key(self, key):
        """Convert indexing key to a hashable cache key."""
        if isinstance(key, tuple):
            return tuple(
                slice(s.start, s.stop, s.step) if isinstance(s, slice)
                else s if isinstance(s, (int, type(Ellipsis)))
                else tuple(s.tolist()) if isinstance(s, torch.Tensor)
                else s
                for s in key
            )
        elif isinstance(key, slice):
            return slice(key.start, key.stop, key.step)
        elif isinstance(key, torch.Tensor):
            return tuple(key.tolist())
        else:
            return key
    
    def _evict_cache(self):
        """Remove oldest cache entries to free memory."""
        # Keep only the most recent half of cache entries
        keys_to_remove = list(self._cache_keys)[:len(self._cache_keys) // 2]
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._cache_keys.discard(key)
    
    def _evaluate_subset(self, coord_subset: torch.Tensor) -> torch.Tensor:
        """Evaluate deformation field for a subset of coordinates."""
        original_shape = coord_subset.shape
        coord_flat = coord_subset.reshape(-1, 3)
        
        if coord_flat.shape[0] == 0:
            return torch.empty(original_shape[:-1] + (2,), 
                             device=self.device, 
                             dtype=self.dtype)
        
        shifts_flat = self.deformation_field(coord_flat)
        return shifts_flat.reshape(original_shape[:-1] + (2,))
    
    def _evaluate_subset_batched(self, coord_subset: torch.Tensor) -> torch.Tensor:
        """Evaluate deformation field for a subset using batching."""
        original_shape = coord_subset.shape
        coord_flat = coord_subset.reshape(-1, 3)
        n_coords = coord_flat.shape[0]
        
        batch_results = []
        for start_idx in range(0, n_coords, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_coords)
            batch_coords = coord_flat[start_idx:end_idx]
            batch_shifts = self.deformation_field(batch_coords)
            batch_results.append(batch_shifts)
            
            # Clear cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        all_shifts = torch.cat(batch_results, dim=0)
        return all_shifts.reshape(original_shape[:-1] + (2,))
    
    def random_subset(self, n_coords: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get shifts for a random subset of coordinates.
        
        Parameters
        ----------
        n_coords: int
            Number of random coordinates to evaluate
            
        Returns
        -------
        shifts: torch.Tensor
            Shifts at the random coordinates
        coordinates: torch.Tensor
            The selected coordinates
        """
        # Flatten coordinates for easier random sampling
        coord_flat = self.coordinates.reshape(-1, 3)
        total_coords = coord_flat.shape[0]
        
        if n_coords >= total_coords:
            # Return all coordinates
            indices = torch.arange(total_coords, device=self.device)
        else:
            # Random sampling
            indices = torch.randperm(total_coords, device=self.device)[:n_coords]
        
        selected_coords = coord_flat[indices]
        shifts = self._evaluate_subset(selected_coords)
        
        return shifts, selected_coords
    
    def evaluate_all(self) -> torch.Tensor:
        """
        Evaluate shifts for all coordinates (equivalent to non-lazy version).
        Uses batching to control memory usage.
        """
        return self._evaluate_subset_batched(self.coordinates)
    
    def clear_cache(self):
        """Clear the internal cache to free memory."""
        self._cache.clear()
        self._cache_keys.clear()
    
    def __repr__(self) -> str:
        return (f"LazyDeformationGridEvaluator(coord_shape={self.coord_shape}, "
                f"shifts_shape={self.shape}, "
                f"batch_size={self.batch_size}, "
                f"cached_items={len(self._cache)})")


def evaluate_deformation_grid_lazy(
    deformation_grid: torch.Tensor,  # (2, nt, nh, nw) or (nt, nh, nw)
    coordinates: torch.Tensor,  # (..., 3)
    batch_size: int = 50000,
) -> LazyDeformationGridEvaluator:
    """
    Create a lazy deformation grid evaluator that computes shifts on-demand.
    
    This is memory-efficient for cases where you have many coordinates but only
    need to evaluate a subset at a time, such as in optimization loops.
    
    Parameters
    ----------
    deformation_grid: torch.Tensor
        The deformation grid data. Shape (2, nt, nh, nw) or (nt, nh, nw).
    coordinates: torch.Tensor
        (..., 3) array of tyx coordinates where shifts should be evaluated.
    batch_size: int
        Batch size for processing coordinates to control memory usage.
        
    Returns
    -------
    lazy_evaluator: LazyDeformationGridEvaluator
        Lazy evaluator object that computes shifts on-demand.
        
    Examples
    --------
    >>> # Create some dummy data
    >>> deformation_grid = torch.randn(2, 10, 5, 5)
    >>> coords = torch.randn(1000, 100, 3)  # Large coordinate array
    >>> 
    >>> # Create lazy evaluator - no computation happens here
    >>> lazy_eval = evaluate_deformation_grid_lazy(deformation_grid, coords)
    >>> 
    >>> # Only evaluate a subset (memory efficient)
    >>> subset_shifts = lazy_eval[0:10, 0:20]  # Only computes 10x20 shifts
    >>> 
    >>> # Or use random subsets in optimization loops
    >>> for i in range(n_iterations):
    ...     shifts, coords_subset = lazy_eval.random_subset(100)
    ...     # Use shifts and coords_subset in your optimization
    """
    return LazyDeformationGridEvaluator(
        deformation_grid=deformation_grid,
        coordinates=coordinates,
        batch_size=batch_size,
    )
