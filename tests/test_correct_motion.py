"""Tests for motion correction functions."""

import pytest
import torch

from torch_motion_correction.correct_motion import (
    correct_motion,
    correct_motion_fast,
    correct_motion_slow,
)


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    # Create a simple test image
    t, h, w = 5, 64, 64
    image = torch.zeros((t, h, w))
    # Add a simple pattern
    for frame_idx in range(t):
        y_center = h // 2
        x_center = w // 2
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        dist_sq = (y - y_center) ** 2 + (x - x_center) ** 2
        image[frame_idx] = torch.exp(-dist_sq / (2 * 10**2))
    return image


@pytest.fixture
def sample_deformation_field():
    """Create a sample deformation field for testing."""
    t = 5
    # Create a simple deformation field with small shifts
    deformation_field = torch.zeros((2, t, 2, 2))
    # Add some small shifts
    for frame_idx in range(t):
        deformation_field[0, frame_idx, :, :] = frame_idx * 0.1  # y shift
        deformation_field[1, frame_idx, :, :] = frame_idx * 0.05  # x shift
    return deformation_field


@pytest.fixture
def sample_single_patch_deformation_field():
    """Create a single patch deformation field for fast correction."""
    t = 5
    # Single patch deformation field: (2, t, 1, 1)
    deformation_field = torch.zeros((2, t, 1, 1))
    for frame_idx in range(t):
        deformation_field[0, frame_idx, 0, 0] = frame_idx * 0.1  # y shift
        deformation_field[1, frame_idx, 0, 0] = frame_idx * 0.05  # x shift
    return deformation_field


@pytest.fixture
def pixel_spacing():
    """Pixel spacing in Angstroms."""
    return 1.0


class TestCorrectMotion:
    """Tests for correct_motion function."""

    def test_basic_functionality(
        self,
        sample_image,
        sample_deformation_field,
        pixel_spacing,
    ):
        """Test basic motion correction."""
        corrected = correct_motion(
            image=sample_image,
            deformation_grid=sample_deformation_field,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        # Check output shape matches input
        assert corrected.shape == sample_image.shape
        assert isinstance(corrected, torch.Tensor)

    def test_different_grid_types(
        self,
        sample_image,
        sample_deformation_field,
        pixel_spacing,
    ):
        """Test different grid types."""
        for grid_type in ["catmull_rom", "bspline"]:
            corrected = correct_motion(
                image=sample_image,
                deformation_grid=sample_deformation_field,
                pixel_spacing=pixel_spacing,
                grid_type=grid_type,
                device=torch.device("cpu"),
            )
            assert corrected.shape == sample_image.shape

    def test_grad_flag(self, sample_image, sample_deformation_field, pixel_spacing):
        """Test grad flag."""
        corrected = correct_motion(
            image=sample_image,
            deformation_grid=sample_deformation_field,
            pixel_spacing=pixel_spacing,
            grad=False,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # When grad=False, output should be detached
        assert not corrected.requires_grad

    def test_different_devices(
        self, sample_image, sample_deformation_field, pixel_spacing
    ):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            corrected = correct_motion(
                image=sample_image,
                deformation_grid=sample_deformation_field,
                pixel_spacing=pixel_spacing,
                device=torch.device("cuda"),
            )
            assert corrected.device.type == "cuda"
            assert corrected.shape == sample_image.shape
        else:
            pytest.skip("CUDA not available")

    def test_zero_deformation_field(self, sample_image, pixel_spacing):
        """Test with zero deformation field (should return original image)."""
        t, h, w = sample_image.shape
        zero_field = torch.zeros((2, t, 2, 2))
        corrected = correct_motion(
            image=sample_image,
            deformation_grid=zero_field,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # With zero deformation, result should be close to original
        # (allowing for small numerical differences from interpolation)
        assert torch.allclose(corrected, sample_image, atol=0.1)


class TestCorrectMotionFast:
    """Tests for correct_motion_fast function."""

    def test_basic_functionality(
        self, sample_image, sample_single_patch_deformation_field
    ):
        """Test basic fast motion correction."""
        corrected = correct_motion_fast(
            image=sample_image,
            deformation_grid=sample_single_patch_deformation_field,
            device=torch.device("cpu"),
        )
        # Check output shape matches input
        assert corrected.shape == sample_image.shape
        assert isinstance(corrected, torch.Tensor)

    def test_single_patch_requirement(self, sample_image, sample_deformation_field):
        """Test that function requires single patch deformation field."""
        with pytest.raises(ValueError, match="Expected single patch deformation field"):
            correct_motion_fast(
                image=sample_image,
                deformation_grid=sample_deformation_field,  # Not single patch
                device=torch.device("cpu"),
            )

    def test_different_devices(
        self, sample_image, sample_single_patch_deformation_field
    ):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            corrected = correct_motion_fast(
                image=sample_image,
                deformation_grid=sample_single_patch_deformation_field,
                device=torch.device("cuda"),
            )
            assert corrected.device.type == "cuda"
            assert corrected.shape == sample_image.shape
        else:
            pytest.skip("CUDA not available")

    def test_zero_deformation_field(self, sample_image):
        """Test with zero deformation field."""
        t = sample_image.shape[0]
        zero_field = torch.zeros((2, t, 1, 1))
        corrected = correct_motion_fast(
            image=sample_image,
            deformation_grid=zero_field,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # With zero deformation, result should be very close to original
        assert torch.allclose(corrected, sample_image, atol=1e-5)


class TestCorrectMotionSlow:
    """Tests for correct_motion_slow function."""

    def test_basic_functionality(self, sample_image, sample_deformation_field):
        """Test basic slow motion correction."""
        corrected = correct_motion_slow(
            image=sample_image,
            deformation_grid=sample_deformation_field,
            device=torch.device("cpu"),
        )
        # Check output shape matches input
        assert corrected.shape == sample_image.shape
        assert isinstance(corrected, torch.Tensor)

    def test_grad_flag(self, sample_image, sample_deformation_field):
        """Test grad flag."""
        corrected = correct_motion_slow(
            image=sample_image,
            deformation_grid=sample_deformation_field,
            grad=False,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # When grad=False, output should be detached
        assert not corrected.requires_grad

    def test_different_devices(self, sample_image, sample_deformation_field):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            corrected = correct_motion_slow(
                image=sample_image,
                deformation_grid=sample_deformation_field,
                device=torch.device("cuda"),
            )
            assert corrected.device.type == "cuda"
            assert corrected.shape == sample_image.shape
        else:
            pytest.skip("CUDA not available")

    def test_zero_deformation_field(self, sample_image):
        """Test with zero deformation field."""
        t, h, w = sample_image.shape
        zero_field = torch.zeros((2, t, 2, 2))
        corrected = correct_motion_slow(
            image=sample_image,
            deformation_grid=zero_field,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # With zero deformation, result should be close to original
        assert torch.allclose(corrected, sample_image, atol=0.1)


class TestMotionCorrectionIntegration:
    """Integration tests for motion correction workflow."""

    def test_estimate_and_correct_workflow(self, sample_image, pixel_spacing):
        """Test a complete workflow: estimate motion then correct."""
        from torch_motion_correction.estimate_motion_xc import estimate_global_motion

        # Estimate motion
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )

        # Correct motion
        corrected = correct_motion(
            image=sample_image,
            deformation_grid=deformation_field,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )

        # Check that correction produces valid output
        assert corrected.shape == sample_image.shape
        assert torch.isfinite(corrected).all()

    def test_fast_correction_workflow(self, sample_image, pixel_spacing):
        """Test workflow with fast correction."""
        from torch_motion_correction.estimate_motion_xc import estimate_global_motion

        # Estimate motion (produces single patch field)
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )

        # Correct motion using fast method
        corrected = correct_motion_fast(
            image=sample_image,
            deformation_grid=deformation_field,
            device=torch.device("cpu"),
        )

        # Check that correction produces valid output
        assert corrected.shape == sample_image.shape
        assert torch.isfinite(corrected).all()
