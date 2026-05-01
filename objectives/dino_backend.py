"""Shared DINO backend for multiple diversity objectives.

This module provides a shared DINO model and preprocessing pipeline
for DPP, Vendi, and DINO diversity.
"""

import logging
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor


class SharedDINOBackend:
    """Shared DINO model backend for feature extraction.
    
    This class loads the DINO model once and provides feature extraction
    for multiple diversity objectives (DPP, Vendi, DINO).
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: torch.device = torch.device("cuda"),
        cache_dir: str = "./model_cache",
    ):
        """Initialize the shared DINO backend.
        
        Args:
            model_name: HuggingFace model name for DINO
            device: Device to load model on
            cache_dir: Directory to cache downloaded models
        """
        self.device = device
        self.model_name = model_name
        
        logging.info(f"[SharedDINOBackend] Loading DINO model: {model_name}")
        
        # Load DINO model and processor
        self.dino_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        ).to(device)
        self.dino_model.eval()
        
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        # ImageNet normalization constants
        self._img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self._img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        logging.info(f"[SharedDINOBackend] DINO model loaded successfully on {device}")
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINO model.
        
        Args:
            images: Tensor of shape (B, C, H, W) in range [0, 1]
            
        Returns:
            Preprocessed images ready for DINO model
        """
        # Convert [0, 1] to [-1, 1]
        images_dino = images * 2.0 - 1.0
        
        # Resize to 256x256 as expected by DINO
        images_processed = F.interpolate(
            images_dino,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        
        # Convert back to [0, 1] for ImageNet normalization
        images_processed = images_processed * 0.5 + 0.5
        
        # Apply ImageNet normalization
        images_processed = (images_processed - self._img_mean) / self._img_std
        
        return images_processed
    
    def extract_features(self, images: torch.Tensor, normalize: bool = True, use_cls: bool = True) -> torch.Tensor:
        """Extract DINO features from images.
        
        Args:
            images: Tensor of shape (B, C, H, W) in range [0, 1]
            normalize: Whether to L2-normalize the features
            use_cls: If True, return CLS token (B, D). If False, return patch tokens (B, N, D)
            
        Returns:
            If use_cls=True: CLS token embeddings of shape (B, D)
            If use_cls=False: Patch token embeddings of shape (B, N, D) where N is number of patches
        """
        # Preprocess images
        images_processed = self.preprocess_images(images)
        
        # Extract features
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            outputs = self.dino_model(pixel_values=images_processed)
            features = outputs.last_hidden_state  # (B, num_tokens, D)
            
            if use_cls:
                # Extract CLS token (first token)
                embeddings = features[:, 0, :]  # (B, D)
            else:
                # Extract patch tokens (all except CLS)
                embeddings = features[:, 1:, :]  # (B, N, D)
            
            # Normalize if requested
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def clear_cache(self):
        """Clear any cached data and free GPU memory."""
        torch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        return f"SharedDINOBackend(model={self.model_name}, device={self.device})"

