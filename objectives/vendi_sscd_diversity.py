"""Vendi diversity objective using SSCD embeddings."""

import torch
import torch.nn.functional as F

from objectives.base_objective import BaseDiversityObjective


class VendiSSCDDiversityObjective(BaseDiversityObjective):
    """Vendi diversity objective using SSCD embeddings (exp of entropy of eigenvalues)."""

    def __init__(
        self,
        weighting: float,
        device: torch.device,
        sscd_backend,
        epsilon: float = 1e-6,
    ):
        """
        Initialize Vendi-SSCD diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
            sscd_backend: SharedSSCDBackend instance for model and preprocessing
            epsilon: Small jitter added to kernel diagonal for numerical stability
        """
        super().__init__("diversity_vendi_sscd", weighting, device)
        if sscd_backend is None:
            raise ValueError("sscd_backend cannot be None for VendiSSCDDiversityObjective")
        self.sscd_backend = sscd_backend
        self.epsilon = epsilon

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute Vendi diversity score using SSCD embeddings.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Vendi diversity score
        """
        batch_size = images.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)

        # Extract SSCD features from shared backend
        # SSCD features are already L2-normalized
        embeddings = self.sscd_backend.extract_features(images, normalize=True)

        # Compute similarity kernel (cosine similarity)
        # Since embeddings are L2-normalized, this is just the dot product
        kernel = embeddings @ embeddings.T  # [B, B]
        
        # Ensure symmetry
        kernel = (kernel + kernel.T) * 0.5
        
        # Add small jitter for numerical stability
        identity = torch.eye(batch_size, device=self.device, dtype=kernel.dtype)
        kernel = kernel + self.epsilon * identity

        # Compute eigenvalues (convert to float32 if needed, as eigvalsh doesn't support Half/BFloat16 on CUDA)
        kernel_for_eig = kernel.float() if kernel.dtype in (torch.float16, torch.bfloat16) else kernel
        eigenvalues = torch.linalg.eigvalsh(kernel_for_eig)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        
        # Compute Vendi score: exp(entropy of normalized eigenvalues)
        total = eigenvalues.sum()
        if total <= 0:
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)
        
        # Normalize eigenvalues to get probability distribution
        probs = eigenvalues / total
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-12)).sum()
        
        # Vendi score is exp(entropy)
        vendi_score = torch.exp(entropy)
        
        return vendi_score

