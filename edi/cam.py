"""Class Activation Mapping: Grad-CAM, Score-CAM, and a real stability measure.

The published pipeline uses Score-CAM. Grad-CAM is kept because it is roughly
two orders of magnitude cheaper and is the right choice for iterating; both
return a normalised heatmap on the same scale so downstream code is agnostic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import DysgraphiaClassifier


def _normalise(cam: torch.Tensor) -> torch.Tensor:
    cam = torch.relu(cam)
    flat = cam.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def generate_gradcam(
    model: DysgraphiaClassifier,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Grad-CAM: activations weighted by global-average-pooled gradients.

    Returns ``None`` when the hooks did not capture anything, rather than
    inventing a uniform heatmap that would silently pollute the feature stage.
    """
    model.eval()
    model.reset_caches()

    input_tensor = input_tensor.to(device)
    output = model(input_tensor)

    model.zero_grad(set_to_none=True)
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1.0
    output.backward(gradient=one_hot, retain_graph=True)

    gradients, features = model.gradients, model.features
    model.reset_caches()

    if gradients is None or features is None:
        return None

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    return _normalise(cam)


def generate_scorecam(
    model: DysgraphiaClassifier,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    num_channels: int = 32,
    batch_size: int = 16,
) -> Optional[torch.Tensor]:
    """Score-CAM: gradient-free, weights each channel by its masked confidence.

    Costs ``num_channels`` extra forward passes per image. The weight for a
    channel is the confidence *gain* over the unmasked baseline, clipped at
    zero, so channels that do not support the target class cannot subtract
    evidence they never contributed.
    """
    model.eval()
    model.reset_caches()
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        features = model.features
        if features is None:
            model.reset_caches()
            return None

        baseline = torch.softmax(output, dim=1)[0, target_class].item()

        channels = features.shape[1]
        channels_to_use = min(channels, num_channels)
        input_h, input_w = input_tensor.shape[2:]

        upsampled = F.interpolate(features[:, :channels_to_use], size=(input_h, input_w), mode="bilinear", align_corners=False)

        flat = upsampled.flatten(2)
        minimum = flat.min(dim=2).values.unsqueeze(2)
        maximum = flat.max(dim=2).values.unsqueeze(2)
        normalised = (flat - minimum) / (maximum - minimum + 1e-8)
        normalised = normalised.view(1, channels_to_use, input_h, input_w)

        gains: List[float] = []
        for start in range(0, channels_to_use, batch_size):
            stop = min(start + batch_size, channels_to_use)
            masks = normalised[0, start:stop].unsqueeze(1)
            masked_input = input_tensor.expand(masks.shape[0], -1, -1, -1) * masks
            masked_output = model(masked_input)
            confidence = torch.softmax(masked_output, dim=1)[:, target_class]
            gains.extend((confidence - baseline).clamp_min(0.0).tolist())

        weights = torch.tensor(gains, device=device, dtype=features.dtype).view(1, -1, 1, 1)
        cam = (weights * features[:, :channels_to_use]).sum(dim=1, keepdim=True)

    model.reset_caches()
    return _normalise(cam)


def generate_cam(
    model: DysgraphiaClassifier,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
    method: str = "scorecam",
    num_channels: int = 32,
) -> Optional[torch.Tensor]:
    """Dispatch to the requested CAM method."""
    if method == "scorecam":
        return generate_scorecam(model, input_tensor, target_class, device, num_channels=num_channels)
    if method == "gradcam":
        return generate_gradcam(model, input_tensor, target_class, device)
    raise ValueError(f"Unknown CAM method: {method!r}. Use 'scorecam' or 'gradcam'.")


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> Optional[float]:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return None
    return intersection / union


def cam_stability_iou(
    models: List[DysgraphiaClassifier],
    images: List[torch.Tensor],
    device: torch.device,
    method: str = "scorecam",
    threshold: float = 0.5,
    num_channels: int = 32,
) -> Dict[str, object]:
    """Measure heatmap agreement between independently trained models.

    This is the quantity the paper refers to as CAM stability. Earlier versions
    of this pipeline returned a constant ``0.5`` in place of it; this function
    computes it, and reports ``None`` when it cannot rather than substituting a
    placeholder. Requires at least two models.
    """
    if len(models) < 2:
        return {"mean_iou": None, "n_images": 0, "n_pairs": 0, "note": "fewer than 2 models supplied"}

    pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
    per_image: List[float] = []
    skipped = 0

    for image in images:
        heatmaps = []
        for model in models:
            with torch.no_grad():
                pred = int(model(image.to(device)).argmax(dim=1).item())
            cam = generate_cam(model, image, pred, device, method=method, num_channels=num_channels)
            if cam is None:
                heatmaps = []
                break
            heatmaps.append(cam.detach().cpu().numpy().squeeze() > threshold)

        if len(heatmaps) != len(models):
            skipped += 1
            continue

        ious = [v for v in (_iou(heatmaps[a], heatmaps[b]) for a, b in pairs) if v is not None]
        if ious:
            per_image.append(float(np.mean(ious)))

    return {
        "mean_iou": float(np.mean(per_image)) if per_image else None,
        "std_iou": float(np.std(per_image)) if per_image else None,
        "n_images": len(per_image),
        "n_pairs": len(pairs),
        "skipped": skipped,
        "method": method,
        "threshold": threshold,
    }
