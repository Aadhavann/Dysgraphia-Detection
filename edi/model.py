"""CNN backbone with the hooks CAM needs."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import FEATURE_DIMS


def _load_backbone(model_type: str, pretrained: bool) -> nn.Module:
    """Build a backbone, preferring the modern ``weights=`` API."""
    if model_type == "resnet18":
        from torchvision.models import resnet18

        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights

                return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except ImportError:
                return resnet18(pretrained=True)
        return resnet18(weights=None)

    if model_type == "efficientnet_b0":
        from torchvision.models import efficientnet_b0

        if pretrained:
            try:
                from torchvision.models import EfficientNet_B0_Weights

                return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            except ImportError:
                return efficientnet_b0(pretrained=True)
        return efficientnet_b0(weights=None)

    raise ValueError(f"Unsupported model_type: {model_type!r}. Use 'resnet18' or 'efficientnet_b0'.")


class DysgraphiaClassifier(nn.Module):
    """Two-class classifier that caches the last conv layer's activations.

    ``features`` holds the forward activations and ``gradients`` the backward
    gradients at ``target_layer``, which is what Grad-CAM consumes. Score-CAM
    only needs ``features``.
    """

    def __init__(self, model_type: str = "resnet18", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if model_type not in FEATURE_DIMS:
            raise ValueError(f"Unsupported model_type: {model_type!r}")

        self.model_type = model_type
        self.feature_dim = FEATURE_DIMS[model_type]
        self.backbone = _load_backbone(model_type, pretrained)

        if model_type == "resnet18":
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            self.target_layer = self.backbone.layer4
        else:
            self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
            self.target_layer = self.backbone.features

        self.features: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.features = output

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        # register_backward_hook is deprecated; full_backward_hook is the
        # supported replacement and fires reliably for these backbones.
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba_positive(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probability of the dysgraphia class."""
        self.eval()
        return torch.softmax(self.forward(x), dim=1)[:, 1]

    def reset_caches(self) -> None:
        self.features = None
        self.gradients = None


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe(model: nn.Module) -> List[str]:
    total, trainable = count_parameters(model)
    return [
        f"backbone: {model.model_type}",
        f"cam target layer: {type(model.target_layer).__name__}",
        f"feature dim: {model.feature_dim}",
        f"parameters: {total:,} total / {trainable:,} trainable",
    ]
