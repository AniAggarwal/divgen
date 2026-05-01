import torch

from objectives.base_objective import BaseDiversityObjective


class VendiDiversityObjective(BaseDiversityObjective):
    """Vendi diversity objective using DINO CLS embeddings (exp of entropy of eigenvalues)."""

    def __init__(
        self,
        weighting: float,
        device: torch.device,
        dino_backend,
        epsilon: float = 1e-6,
    ):
        """
        Initialize Vendi diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
            dino_backend: SharedDINOBackend instance for model and preprocessing
            epsilon: Small jitter added to kernel diagonal for numerical stability
        """
        super().__init__("diversity_vendi", weighting, device)
        self.dino_backend = dino_backend
        self.epsilon = epsilon

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute Vendi diversity score (exp of entropy of eigenvalues).
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Vendi diversity score
        """
        batch_size = images.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)

        # Extract DINO CLS token features from shared backend
        cls_embeddings = self.dino_backend.extract_features(images, normalize=True, use_cls=True)

        # Compute similarity kernel (cosine similarity)
        kernel = cls_embeddings @ cls_embeddings.T  # [B, B]
        
        # Ensure symmetry
        kernel = (kernel + kernel.T) * 0.5
        
        # Add small jitter for numerical stability
        identity = torch.eye(batch_size, device=self.device, dtype=kernel.dtype)
        kernel = kernel + self.epsilon * identity

        # Compute eigenvalues (convert to float32 if needed, as eigvalsh doesn't support Half on CUDA)
        kernel_for_eig = kernel.float() if kernel.dtype == torch.float16 else kernel
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

