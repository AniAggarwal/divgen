import shutil
from pathlib import Path

import huggingface_hub
import torch


def _ensure_hpsv2_clip_vocab() -> None:
    """The hpsv2==1.2.0 wheel ships its vendored open_clip without the BPE
    vocab file, so importing it raises FileNotFoundError. The file is
    byte-identical to the one shipped by openai-clip, so copy it across.
    Must run before `hpsv2.src.open_clip` is imported, since that subpackage
    reads the vocab at import time."""
    import hpsv2

    target = Path(hpsv2.__file__).parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
    if target.exists():
        return
    import clip
    source = Path(clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    shutil.copyfile(source, target)


_ensure_hpsv2_clip_vocab()

from hpsv2.src.open_clip import create_model, get_tokenizer
from hpsv2.src.open_clip.model import convert_weights_to_lp

from rewards.base_reward import BaseRewardLoss


class HPSLoss(BaseRewardLoss):
    """HPS reward loss function for optimization."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
    ):
        precision_str = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }[dtype]
        self.hps_model = create_model(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision=precision_str,
            device=device,
            cache_dir=cache_dir,
        )
        checkpoint_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", "HPS_v2.1_compressed.pt", cache_dir=cache_dir
        )
        self.hps_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)["state_dict"]
        )
        self.hps_tokenizer = get_tokenizer("ViT-H-14")
        self.device = device
        self.hps_model = self.hps_model.to(device)
        if dtype != torch.float32:
            convert_weights_to_lp(self.hps_model, dtype=dtype)
        self.hps_model.eval()
        self.freeze_parameters(self.hps_model.parameters())
        super().__init__("HPS", weighting)
        self.hps_model.set_grad_checkpointing(True)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        hps_image_features = self.hps_model.encode_image(image)
        return hps_image_features

    def get_text_features(self, prompt: str) -> torch.Tensor:
        hps_text = self.hps_tokenizer(prompt).to(self.device)
        hps_text_features = self.hps_model.encode_text(hps_text)
        return hps_text_features

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        logits_per_image = image_features @ text_features.T
        hps_loss = 1 - torch.diagonal(logits_per_image).mean()
        return hps_loss

    def get_per_sample_scores(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Return per-sample HPS scores (not averaged). Higher = better quality."""
        logits_per_image = image_features @ text_features.T
        return torch.diagonal(logits_per_image)
