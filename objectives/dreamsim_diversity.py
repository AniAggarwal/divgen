import torch

from objectives.base_objective import BaseDiversityObjective


class DreamSimDiversityObjective(BaseDiversityObjective):
    """DreamSim-based perceptual diversity objective."""

    def __init__(
        self,
        weighting: float,
        device: torch.device,
        dreamsim_backend=None,
        cache_dir: str = None,
        backbone: str = "dino_vitb16",
    ):
        """
        Initialize DreamSim diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
            dreamsim_backend: Optional shared DreamSim backend to reuse
            cache_dir: Cache directory for model weights (only used if backend not provided)
            backbone: Backbone architecture ('dino_vitb16', 'clip_vitb32', 'open_clip_vitb32', 'ensemble')
        """
        super().__init__("diversity_dreamsim", weighting, device)
        
        if dreamsim_backend is not None:
            # Use shared backend
            self.dreamsim_backend = dreamsim_backend
            self.owns_backend = False
        else:
            # Create own backend (legacy path)
            from objectives.dreamsim_backend import SharedDreamSimBackend
            self.dreamsim_backend = SharedDreamSimBackend(
                device=device, 
                cache_dir=cache_dir,
                backbone=backbone
            )
            self.owns_backend = True

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise DreamSim distances for all image pairs.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Mean DreamSim distance across all pairs
        """
        batch_size = images.shape[0]
        
        # Try feature caching approach first
        features = self.dreamsim_backend.extract_features(images)
        
        if features is not None:
            # Use cached features for pairwise distance computation
            pairwise_scores = []
            for idx_a in range(batch_size - 1):
                for idx_b in range(idx_a + 1, batch_size):
                    dist = self.dreamsim_backend.compute_distance_from_features(
                        features[idx_a : idx_a + 1],
                        features[idx_b : idx_b + 1],
                    )
                    pairwise_scores.append(dist.view(-1))
        else:
            # Fallback: compute full forward pass for each pair
            pairwise_scores = []
            for idx_a in range(batch_size - 1):
                for idx_b in range(idx_a + 1, batch_size):
                    dist = self.dreamsim_backend.compute_distance(
                        images[idx_a : idx_a + 1],
                        images[idx_b : idx_b + 1],
                    )
                    pairwise_scores.append(dist.view(-1))
        
        if pairwise_scores:
            diversity_score = torch.cat(pairwise_scores).mean()
            return diversity_score
        else:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

