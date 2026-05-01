import torch
import torch.nn.functional as F

from objectives.base_objective import BaseDiversityObjective


class DPPPatchDiversityObjective(BaseDiversityObjective):
    """DPP diversity over per-patch DINO features.

    Differs from DPPDiversityObjective in two ways:
    1. Uses DINO patch features (use_cls=False) instead of the CLS token.
    2. Computes one logdet(I + K_p) per patch (where K_p is the BxB cosine
       kernel of normalized patch-p features across the batch), then averages
       across patches. This matches the patchwise diversity statistic in
       Eq. 3 of the paper (v_B = (1/P) sum_p ...).
    """

    def __init__(
        self,
        weighting: float,
        device: torch.device,
        dino_backend,
        epsilon: float = 1e-6,
    ):
        super().__init__("diversity_dpp_patch", weighting, device)
        self._last_raw_score: float = 0.0
        if dino_backend is None:
            raise ValueError("DINO backend required for DPPPatchDiversityObjective.")
        self.feature_backend = dino_backend
        self.epsilon = epsilon

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        if batch_size <= 1:
            self._last_raw_score = 0.0
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)

        # Patch features: (B, num_patches, hidden_dim), L2-normalized along hidden_dim.
        features = self.feature_backend.extract_features(
            images, normalize=False, use_cls=False
        )
        features = F.normalize(features, p=2, dim=-1)

        # Per-patch BxB cosine kernels: (num_patches, B, B).
        # features[:, p, :] @ features[:, p, :].T for each p, vectorized via einsum.
        kernels = torch.einsum("bpd,cpd->pbc", features, features)
        # Symmetrize.
        kernels = 0.5 * (kernels + kernels.transpose(-1, -2))

        identity = torch.eye(batch_size, device=self.device, dtype=kernels.dtype)
        kernels = kernels + self.epsilon * identity  # broadcasts over patch dim
        regularized = identity + kernels  # (num_patches, B, B)

        sign, logdet = torch.linalg.slogdet(regularized)
        # Drop patches with non-positive determinant (numerical pathologies).
        valid = sign > 0
        if valid.any():
            mean_logdet = logdet[valid].mean()
        else:
            mean_logdet = torch.tensor(0.0, device=self.device, dtype=images.dtype)

        self._last_raw_score = mean_logdet.item()
        max_score = batch_size * torch.log(torch.tensor(2.0, device=self.device, dtype=mean_logdet.dtype))
        return mean_logdet / max_score
