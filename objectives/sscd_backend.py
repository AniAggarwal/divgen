"""Shared SSCD backend for diversity objectives.

This module provides a shared SSCD (Self-Supervised Copy Detection) model 
for computing perceptual similarity and diversity.
"""

import logging
import torch
import torch.nn as nn


class SharedSSCDBackend:
    """Shared SSCD model backend for perceptual similarity computation.
    
    This class loads the SSCD model once and provides feature extraction
    for diversity objectives. SSCD produces L2-normalized 512-dimensional embeddings.
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        cache_dir: str = "./cache",
        model_name: str = "sscd_disc_mixup",
    ):
        """Initialize the shared SSCD backend.
        
        Args:
            device: Device to load model on
            cache_dir: Directory to cache downloaded models
            model_name: SSCD model variant to use (default: sscd_disc_mixup)
        """
        self.device = device
        self.model_name = model_name
        
        logging.info(f"[SharedSSCDBackend] Loading SSCD model: {model_name}")
        
        # Download and load the torchscript model
        import os
        import urllib.request
        
        model_urls = {
            "sscd_disc_mixup": "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
        }
        
        if model_name not in model_urls:
            raise ValueError(f"Unknown SSCD model: {model_name}. Choose from {list(model_urls.keys())}")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, f"{model_name}.torchscript.pt")
        
        # Download if not cached
        if not os.path.exists(model_path):
            logging.info(f"[SharedSSCDBackend] Downloading {model_name} to {model_path}")
            urllib.request.urlretrieve(model_urls[model_name], model_path)
            logging.info(f"[SharedSSCDBackend] Download complete")
        else:
            logging.info(f"[SharedSSCDBackend] Using cached model from {model_path}")
        
        # Load the torchscript model
        # TorchScript models work best with float32
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        
        # Ensure model is in float32 (TorchScript SSCD models expect this)
        self.model = self.model.float()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # SSCD preprocessing (from their documentation)
        from torchvision import transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
        )
        
        logging.info(f"[SharedSSCDBackend] SSCD model ({model_name}) loaded successfully on {device}")
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SSCD model.
        
        Args:
            images: Batch of images with shape (B, C, H, W) in range [0, 1]
            
        Returns:
            Preprocessed images in float32
        """
        # Ensure float32 for TorchScript model compatibility
        images = images.float()
        
        # SSCD expects normalized images
        # Images are already in [0, 1] range, just need to normalize
        normalized = self.normalize(images)
        
        # Ensure output is float32
        return normalized.float()
    
    def extract_features(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Extract SSCD features from images.
        
        Args:
            images: Batch of images with shape (B, C, H, W) in range [0, 1]
            normalize: Whether to L2 normalize the features (SSCD already does this)
            
        Returns:
            Feature embeddings of shape (B, 512) or (B, 1024) for large model
        """
        # Ensure images are on the correct device and dtype
        # SSCD TorchScript models typically expect float32
        original_dtype = images.dtype
        images = images.to(device=self.device, dtype=torch.float32)
        
        # Preprocess images
        preprocessed = self.preprocess(images)
        
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            # Extract features
            features = self.model(preprocessed)
        
        # Convert back to original dtype if needed
        if original_dtype != torch.float32:
            features = features.to(dtype=original_dtype)
        
        # SSCD models already output L2-normalized features
        # But we can optionally normalize again for safety
        if normalize:
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features
    
    def compute_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSCD similarity between two images.
        
        Args:
            img1: First image tensor of shape (B1, C, H, W) in range [0, 1]
            img2: Second image tensor of shape (B2, C, H, W) in range [0, 1]
            
        Returns:
            Similarity scores (cosine similarity) of shape (B1, B2)
        """
        feat1 = self.extract_features(img1, normalize=True)
        feat2 = self.extract_features(img2, normalize=True)
        
        # Cosine similarity (features are already normalized)
        similarity = feat1 @ feat2.T
        
        return similarity
    
    def clear_cache(self):
        """Clear any cached data and free GPU memory."""
        torch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        return f"SharedSSCDBackend(model={self.model_name}, device={self.device})"

