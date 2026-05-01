"""Shared DreamSim backend for diversity objectives.

This module provides a shared DreamSim model for computing perceptual similarity.
"""

import logging
import torch


class SharedDreamSimBackend:
    """Shared DreamSim model backend for perceptual similarity computation.
    
    This class loads the DreamSim model once and provides similarity computation
    for diversity objectives. Supports feature caching to avoid redundant forward passes.
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        cache_dir: str = "./model_cache",
        backbone: str = "dino_vitb16",
    ):
        """Initialize the shared DreamSim backend.
        
        Args:
            device: Device to load model on
            cache_dir: Directory to cache downloaded models
            backbone: Backbone architecture to use. Options: 'dino_vitb16', 'clip_vitb32', 'open_clip_vitb32', 'ensemble'
        """
        self.device = device
        self.backbone = backbone
        
        logging.info(f"[SharedDreamSimBackend] Loading DreamSim model with backbone: {backbone}")
        
        try:
            from dreamsim import dreamsim
        except ImportError:
            raise ImportError(
                "DreamSim backend requested but 'dreamsim' package is not installed. "
                "Please install it with: pip install dreamsim"
            )
        
        # Load DreamSim model with specified backbone
        self.dreamsim_model, self.preprocess = dreamsim(
            pretrained=True, 
            cache_dir=cache_dir,
            dreamsim_type=backbone
        )
        self.dreamsim_model = self.dreamsim_model.to(device)
        self.dreamsim_model.eval()
        
        # Freeze parameters
        for param in self.dreamsim_model.parameters():
            param.requires_grad = False
        
        logging.info(f"[SharedDreamSimBackend] DreamSim model ({backbone}) loaded successfully on {device}")
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DreamSim features from images.
        
        Args:
            images: Batch of images with shape (B, C, H, W) in range [0, 1]
            
        Returns:
            Feature embeddings of shape (B, D)
        """
        # DreamSim expects images in [0, 1] range as float32
        images_f32 = images.to(torch.float32)
        
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            # Check if model has embed method for feature extraction
            if hasattr(self.dreamsim_model, 'embed'):
                features = self.dreamsim_model.embed(images_f32)
            else:
                # Fallback: use the model's internal feature extractor if available
                # This depends on the DreamSim implementation
                logging.warning("[SharedDreamSimBackend] Model does not have 'embed' method, using full forward pass")
                # For now, return None to indicate feature caching is not supported
                return None
        
        return features
    
    def compute_distance(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute DreamSim distance between two images.
        
        Args:
            img1: First image tensor of shape (1, C, H, W) in range [0, 1]
            img2: Second image tensor of shape (1, C, H, W) in range [0, 1]
            
        Returns:
            Distance score (scalar tensor)
        """
        # DreamSim expects images in [0, 1] range as float32
        img1_f32 = img1.to(torch.float32)
        img2_f32 = img2.to(torch.float32)
        
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            dist = self.dreamsim_model(img1_f32, img2_f32)
        
        return dist
    
    def compute_distance_from_features(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Compute distance from pre-extracted features.
        
        Args:
            feat1: Feature tensor of shape (D,) or (1, D)
            feat2: Feature tensor of shape (D,) or (1, D)
            
        Returns:
            Distance score (scalar tensor)
        """
        # Check if model has a distance head
        if hasattr(self.dreamsim_model, 'distance'):
            return self.dreamsim_model.distance(feat1, feat2)
        else:
            # Fallback: compute cosine distance
            feat1_norm = feat1 / (feat1.norm(dim=-1, keepdim=True) + 1e-8)
            feat2_norm = feat2 / (feat2.norm(dim=-1, keepdim=True) + 1e-8)
            # Return 1 - cosine similarity as distance
            return 1.0 - (feat1_norm * feat2_norm).sum(dim=-1)
    
    def clear_cache(self):
        """Clear any cached data and free GPU memory."""
        torch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        return f"SharedDreamSimBackend(device={self.device})"

