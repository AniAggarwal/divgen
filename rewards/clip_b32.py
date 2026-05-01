import torch
import torch.nn.functional as F
from transformers import CLIPModel
from transformers.models.clip.modeling_clip import _get_vector_norm

from rewards.base_reward import BaseRewardLoss


class CLIPB32Loss(BaseRewardLoss):
    """CLIP B/32 reward loss function for optimization.

    This uses the base CLIP ViT-B/32 model with manual feature extraction
    similar to the group-inference implementation.
    """

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        tokenizer,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
        )
        self.clip_model = self.clip_model.to(device, dtype=dtype)
        self.clip_model.eval()
        self.freeze_parameters(self.clip_model.parameters())
        super().__init__("CLIP_b32", weighting)
        self.clip_model.gradient_checkpointing_enable()

        # Store CLIP normalization constants
        self._img_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        self._img_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract CLIP image features.

        Args:
            image: Image tensor in range [-1, 1]

        Returns:
            Normalized image embeddings
        """
        # Resize to CLIP input size
        image_resized = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)

        # Re-normalize from [-1, 1] to CLIP normalization
        image_resized = image_resized * 0.5 + 0.5  # Convert to [0, 1]
        image_resized = torch.clamp(image_resized, min=0, max=1)
        image_resized = (image_resized - self._img_mean) / self._img_std

        # Extract features
        vision_outputs = self.clip_model.vision_model(
            pixel_values=image_resized,
            output_attentions=False,
            output_hidden_states=False
        )
        image_embeds = self.clip_model.visual_projection(vision_outputs.pooler_output)

        # Normalize embeddings with epsilon to prevent division by zero
        image_embeds_norm = _get_vector_norm(image_embeds)
        image_embeds = image_embeds / torch.clamp(image_embeds_norm, min=1e-8)

        return image_embeds

    def get_text_features(self, prompt: str) -> torch.Tensor:
        """Extract CLIP text features.

        Args:
            prompt: Text prompt

        Returns:
            Normalized text embeddings
        """
        # Check cache first (from base class)
        if prompt in self._text_feature_cache:
            return self._text_feature_cache[prompt]

        # Tokenize and encode
        text_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            max_length=77,
            truncation=True
        ).to(self.device)

        # Extract text features
        text_outputs = self.clip_model.text_model(**text_encoding)
        text_embeds = self.clip_model.text_projection(text_outputs.pooler_output)

        # Normalize embeddings with epsilon to prevent division by zero
        text_embeds_norm = _get_vector_norm(text_embeds)
        text_embeds = text_embeds / torch.clamp(text_embeds_norm, min=1e-8)

        # Cache the result (using base class cache)
        self._text_feature_cache[prompt] = text_embeds

        return text_embeds

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute CLIP loss.

        This follows the group-inference implementation for similarity computation:
        text_embeds @ image_embeds.T (dot product of normalized embeddings)

        Returns a loss that decreases as similarity increases.
        Uses 1 - similarity so loss is positive and decreases with better alignment.

        Args:
            image_features: Normalized image embeddings
            text_features: Normalized text embeddings

        Returns:
            CLIP loss (lower is better, positive values)
        """
        # Compute cosine similarity (dot product of normalized vectors)
        # Shape: (batch_size, 1) if text is single prompt
        similarity = torch.matmul(text_features, image_features.t()).t()

        # Return 1 - similarity as loss
        # As similarity increases (better alignment), loss decreases
        # Loss ranges from 0 (perfect match) to 2 (opposite vectors)
        clip_loss = 1.0 - similarity.mean()

        return clip_loss

