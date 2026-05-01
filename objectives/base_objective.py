from abc import ABC, abstractmethod

import torch


class BaseDiversityObjective(ABC):
    """
    Base class for diversity objectives implementing differentiable pairwise diversity functions.
    """

    def __init__(self, name: str, weighting: float, device: torch.device):
        self.name = name
        self.weighting = weighting
        self.device = device

    @staticmethod
    def freeze_parameters(params):
        """Freeze model parameters."""
        for param in params:
            param.requires_grad = False

    @abstractmethod
    def compute_pairwise_diversity(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise diversity score for a batch of images.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Average pairwise diversity score (higher = more diverse)
        """
        pass

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss for a batch of images.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Diversity loss (negative of diversity score for maximization)
        """
        if images.shape[0] <= 1:
            # No diversity to compute for single image
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)
        
        diversity_score = self.compute_pairwise_diversity(images)
        # Negative sign because we want to maximize diversity (minimize loss)
        return -diversity_score

