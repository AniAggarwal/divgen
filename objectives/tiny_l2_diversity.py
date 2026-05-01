import torch
import torch.nn.functional as F

from objectives.base_objective import BaseDiversityObjective


# Theoretical maximum L2 distance for 32x32 RGB images (vectors in [0,1])
# Max distance occurs between pure black (0,0,0) and pure white (1,1,1)
# L2_max = sqrt(3 * 32 * 32 * 1^2) = sqrt(3072) ≈ 55.4256
MAX_L2_DISTANCE = float(torch.sqrt(torch.tensor(3 * 32 * 32)).item())


class TinyL2DiversityObjective(BaseDiversityObjective):
    """Tiny L2 diversity objective using L2 distance on downsampled 32x32 images."""

    def __init__(
        self,
        weighting: float,
        device: torch.device,
    ):
        """
        Initialize Tiny L2 diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
        """
        super().__init__("diversity_tiny_l2", weighting, device)

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise diversity using L2 distance on downsampled images.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Mean pairwise L2 distance, normalized to [0, 1]
        """
        N = images.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)

        # Step 1: Downsize images to 32x32
        images_downsampled = F.interpolate(
            images,
            size=(32, 32),
            mode='bilinear',
            align_corners=False
        )

        # Step 2: Flatten to get vectors
        # Shape: (N, C*32*32) = (N, 3072) for RGB images
        vectors = images_downsampled.reshape(N, -1)

        # Step 3: Compute pairwise L2 distances
        # Expand for broadcasting: (N, 1, 3072) and (1, N, 3072)
        vectors_expanded_1 = vectors.unsqueeze(1)  # (N, 1, 3072)
        vectors_expanded_2 = vectors.unsqueeze(0)  # (1, N, 3072)

        # Compute L2 distance: ||v_i - v_j||_2
        # Shape: (N, N)
        distance_matrix = torch.norm(vectors_expanded_1 - vectors_expanded_2, p=2, dim=2)

        # Extract upper triangular part (excluding diagonal) for pairwise distances
        # Create mask for upper triangle
        mask = torch.triu(torch.ones(N, N, device=self.device), diagonal=1).bool()
        pairwise_distances = distance_matrix[mask]

        # Return mean pairwise distance as diversity score, normalized to [0, 1]
        if len(pairwise_distances) > 0:
            diversity_score = pairwise_distances.mean() / MAX_L2_DISTANCE
            return diversity_score
        else:
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)
