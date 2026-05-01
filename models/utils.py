"""Utility functions for model initialization and management."""


def freeze_params(params):
    """Freeze model parameters to prevent gradient updates.
    
    Args:
        params: Iterator of model parameters (e.g., model.parameters())
    """
    for param in params:
        param.requires_grad = False

