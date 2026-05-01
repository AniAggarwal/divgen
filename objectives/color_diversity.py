import torch

from objectives.base_objective import BaseDiversityObjective


# Theoretical maximum L2 distance for color histogram diversity
# Each channel histogram is normalized (sums to 1), max L2 distance per channel is sqrt(2)
# For 3 channels (R, G, B): max_distance = sqrt(2 + 2 + 2) = sqrt(6) ≈ 2.449
MAX_COLOR_HIST_DISTANCE = float(torch.sqrt(torch.tensor(6.0)).item())


class ColorDiversityObjective(BaseDiversityObjective):
    """Color histogram-based diversity objective."""

    def __init__(
        self,
        weighting: float,
        device: torch.device,
    ):
        """
        Initialize Color diversity objective.
        
        Args:
            weighting: Weight for this objective in the total loss
            device: Device for computation
        """
        super().__init__("diversity_color", weighting, device)
        self.n_bins = 32  # Hardcoded number of bins per channel

    def compute_pairwise_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise diversity using color histogram differences.
        
        Args:
            images: Batch of images with shape (B, C, H, W), values in [0, 1]
            
        Returns:
            Mean pairwise diversity score based on color histogram distance
        """
        batch_size = images.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=self.device, dtype=images.dtype)

        # Convert to float32 if needed (torch.histc doesn't support float16/bfloat16)
        original_dtype = images.dtype
        if images.dtype in (torch.float16, torch.bfloat16):
            images = images.float()

        # Compute color histograms for each image
        histograms = []

        for i in range(batch_size):
            img = images[i]  # [C, H, W]

            # Compute histogram for each channel (R, G, B)
            channel_hists = []
            for c in range(3):
                channel = img[c].flatten()  # [H*W]

                # Use standard histogram (no gradients since we're in no_grad context)
                hist = torch.histc(channel, bins=self.n_bins, min=0.0, max=1.0)
                # Normalize
                hist = hist / (hist.sum() + 1e-8)
                channel_hists.append(hist)

            # Concatenate histograms from all channels
            full_hist = torch.cat(channel_hists, dim=0)  # [3 * n_bins]
            histograms.append(full_hist)

        # Stack all histograms
        histograms = torch.stack(histograms, dim=0)  # [B, 3 * n_bins]

        # Compute pairwise L2 distances between histograms
        pairwise_scores = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Compute L2 distance between histograms
                dist = torch.norm(histograms[i] - histograms[j], p=2)
                pairwise_scores.append(dist)

        # Return mean of all pairwise distances, normalized to [0, 1]
        if pairwise_scores:
            diversity_score = torch.stack(pairwise_scores).mean() / MAX_COLOR_HIST_DISTANCE
            # Convert back to original dtype if needed
            if original_dtype == torch.float16:
                diversity_score = diversity_score.half()
            elif original_dtype == torch.bfloat16:
                diversity_score = diversity_score.to(torch.bfloat16)
            return diversity_score
        else:
            return torch.tensor(0.0, device=self.device, dtype=original_dtype)

