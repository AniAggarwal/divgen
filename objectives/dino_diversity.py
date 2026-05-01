import torch
import torch.nn.functional as F

from objectives.base_objective import BaseDiversityObjective


class DINODiversityObjective(BaseDiversityObjective):
    """DINO-based diversity objective using patch-level features."""

    def __init__(
        self,
        weighting: float,
        device: torch.device,
        dino_backend,
    ):
        """
        Initialize DINO diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
            dino_backend: SharedDINOBackend instance for model and preprocessing
        """
        super().__init__("diversity_dino", weighting, device)
        self.dino_backend = dino_backend

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise DINO diversity using patch features.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Mean pairwise diversity score based on DINO features
        """
        # Extract patch features (not CLS token) from shared backend
        all_features = self.dino_backend.extract_features(images, normalize=False, use_cls=False)
        all_features_normed = F.normalize(all_features, p=2, dim=-1)
        
        # Compute pairwise cosine distance matrix
        # Shape: (B, B, num_patches)
        f1 = all_features_normed.unsqueeze(1)  # (B, 1, num_patches, hidden_dim)
        f2 = all_features_normed.unsqueeze(0)  # (1, B, num_patches, hidden_dim)
        
        # Compute 1 - cosine_similarity and average over patches
        score_matrix = (1 - F.cosine_similarity(f1, f2, dim=-1)).mean(dim=-1)  # (B, B)
        
        # Extract upper triangle (excluding diagonal) for pairwise scores
        batch_size = images.shape[0]
        pairwise_scores = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                pairwise_scores.append(score_matrix[i, j])
        
        if pairwise_scores:
            diversity_score = torch.stack(pairwise_scores).mean()
            return diversity_score
        else:
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)

