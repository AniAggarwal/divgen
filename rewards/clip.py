import torch
from transformers import CLIPModel

from rewards.base_reward import BaseRewardLoss


class CLIPLoss(BaseRewardLoss):
    """CLIP reward loss function for optimization."""

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
        self.clip_model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=cache_dir,
        )
        self.clip_model = self.clip_model.to(device, dtype=dtype)
        self.clip_model.eval()
        self.freeze_parameters(self.clip_model.parameters())
        super().__init__("CLIP", weighting)
        self.clip_model.gradient_checkpointing_enable()

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        clip_img_features = self.clip_model.get_image_features(image)
        return clip_img_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        prompt_token = self.tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=77, truncation=True
        ).to(self.device)
        clip_text_features = self.clip_model.get_text_features(**prompt_token)
        return clip_text_features

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        clip_loss = (
            100
            - (image_features @ text_features.T).mean()
            * self.clip_model.logit_scale.exp()
        )
        return clip_loss
