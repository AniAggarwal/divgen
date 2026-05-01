import torch
import torch.nn.functional as F

from objectives.base_objective import BaseDiversityObjective


class DPPDiversityObjective(BaseDiversityObjective):
    """DPP (Determinantal Point Process) diversity objective supporting multiple feature backends."""

    SUPPORTED_BACKENDS = {"dino", "dreamsim"}

    def __init__(
        self,
        weighting: float,
        device: torch.device,
        dino_backend=None,
        dreamsim_backend=None,
        backend_type: str = "dino",
        epsilon: float = 1e-6,
    ):
        """
        Initialize DPP diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
            feature_backend: Backend instance that exposes an extract_features(...) method
            backend_type: Identifier for feature backend ('dino' or 'dreamsim')
            epsilon: Small jitter added to kernel diagonal for numerical stability
        """
        super().__init__("diversity_dpp", weighting, device)
        self._last_raw_score: float = 0.0  # Store raw score for metrics reporting
        if backend_type not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported DPP backend '{backend_type}'. "
                f"Supported backends: {sorted(self.SUPPORTED_BACKENDS)}"
            )
        if backend_type == "dreamsim":
            if dreamsim_backend is None:
                raise ValueError("DreamSim backend requested for DPP but not initialized.")
            self.feature_backend = dreamsim_backend
        else:
            if dino_backend is None:
                raise ValueError("DINO backend requested for DPP but not initialized.")
            self.feature_backend = dino_backend
        self.backend_type = backend_type
        self.epsilon = epsilon

    def _extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract normalized embeddings using the configured backend."""
        if self.backend_type == "dino":
            embeddings = self.feature_backend.extract_features(
                images, normalize=True, use_cls=True
            )
        elif self.backend_type == "dreamsim":
            embeddings = self.feature_backend.extract_features(images)
            if embeddings is None:
                raise RuntimeError(
                    "DreamSim backend did not return embeddings for DPP computation."
                )
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        else:
            raise ValueError(f"Unsupported DPP backend '{self.backend_type}'")
        return embeddings

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized DPP diversity score using log-determinant of similarity kernel.
        
        Returns normalized score [0, 1]. Raw score stored in _last_raw_score for metrics.
        """
        batch_size = images.shape[0]
        if batch_size <= 1:
            self._last_raw_score = 0.0
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)

        cls_embeddings = self._extract_embeddings(images)
        kernel = cls_embeddings @ cls_embeddings.T
        kernel = (kernel + kernel.T) * 0.5
        identity = torch.eye(batch_size, device=self.device, dtype=kernel.dtype)
        kernel = kernel + self.epsilon * identity

        sign, logdet = torch.linalg.slogdet(identity + kernel)
        raw_score = logdet if sign.item() > 0 else torch.tensor(0.0, device=self.device, dtype=images.dtype)
        
        # Store raw score for metrics, return normalized
        self._last_raw_score = raw_score.item()
        max_score = batch_size * torch.log(torch.tensor(2.0, device=self.device, dtype=images.dtype))
        return raw_score / max_score

