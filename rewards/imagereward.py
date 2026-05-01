import ImageReward as RM
import torch

from rewards.base_reward import BaseRewardLoss


class ImageRewardLoss(BaseRewardLoss):
    """Image reward loss for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
    ):
        super().__init__(name="ImageReward", weighting=weighting)
        self.dtype = dtype
        self.imagereward_model = RM.load("ImageReward-v1.0", download_root=cache_dir)
        self.imagereward_model = self.imagereward_model.to(
            device=device, dtype=self.dtype
        )
        self.imagereward_model.eval()
        BaseRewardLoss.freeze_parameters(self.imagereward_model.parameters())

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.imagereward_model.blip.visual_encoder(image)

    def get_text_features(self, prompt: str) -> torch.Tensor:
        return self.imagereward_model.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.imagereward_model.device)

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        # ImageReward fuses image and text via cross-attention, so it can't use
        # the base-class normalize-and-dot-product path.
        image_atts = torch.ones(
            image_features.size()[:-1], dtype=torch.long
        ).to(self.imagereward_model.device)
        text_output = self.imagereward_model.blip.text_encoder(
            text_features.input_ids,
            attention_mask=text_features.attention_mask,
            encoder_hidden_states=image_features,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].to(
            self.imagereward_model.device, dtype=self.dtype
        )
        rewards = self.imagereward_model.mlp(txt_features)
        rewards = (rewards - self.imagereward_model.mean) / self.imagereward_model.std
        return (2 - rewards).mean()

    def __call__(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        image_features = self.get_image_features(image)

        if prompt not in self._text_feature_cache:
            self._text_feature_cache[prompt] = self.get_text_features(prompt)
        text_features = self._text_feature_cache[prompt]

        return self.compute_loss(image_features, text_features)
