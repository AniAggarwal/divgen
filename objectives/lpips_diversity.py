import torch

from objectives.base_objective import BaseDiversityObjective


class LPIPSDiversityObjective(BaseDiversityObjective):
    """LPIPS-based diversity objective for batch of images."""

    def __init__(
        self,
        weighting: float,
        device: torch.device,
        net: str = "vgg",
    ):
        """
        Initialize LPIPS diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
            net: LPIPS network backbone ('alex', 'squeeze', or 'vgg')
        """
        super().__init__("diversity_lpips", weighting, device)
        
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "LPIPS diversity loss requested but the 'lpips' package is not installed. "
                "Please install it with: pip install lpips"
            )
        
        self.lpips = lpips.LPIPS(net=net, verbose=False).to(device)
        self.lpips.eval()
        self.freeze_parameters(self.lpips.parameters())

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise LPIPS distances for all image pairs.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Mean LPIPS distance across all pairs
        """
        # LPIPS expects images in range [-1, 1]
        images_lpips = (images * 2.0 - 1.0).to(torch.float32)
        
        pairwise_scores = []
        batch_size = images_lpips.shape[0]
        
        # Compute all pairwise distances
        for idx_a in range(batch_size - 1):
            for idx_b in range(idx_a + 1, batch_size):
                dist = self.lpips(
                    images_lpips[idx_a : idx_a + 1],
                    images_lpips[idx_b : idx_b + 1],
                )
                pairwise_scores.append(dist.view(-1))
        
        if pairwise_scores:
            diversity_score = torch.cat(pairwise_scores).mean()
            return diversity_score
        else:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

