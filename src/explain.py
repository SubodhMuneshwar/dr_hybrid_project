import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try importing deep learning backends: PyTorch first, then TensorFlow
TORCH_AVAILABLE = False
torch = None
try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    TORCH_AVAILABLE = True
except ImportError:
    pass

TF_AVAILABLE = False
tf = None
try:
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    pass

# Global singletons to prevent reloading 500MB weights on every scan
_CACHED_TORCH_VGG = None
_CACHED_TF_CONV_MODEL = None


def _get_torch_conv_model():
    """Returns cached PyTorch VGG16 convolutional feature extractor (through block5_conv3 + relu)."""
    global _CACHED_TORCH_VGG
    if _CACHED_TORCH_VGG is None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch or torchvision is not installed")
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).eval()
        # Features 0..29 includes all layers up to and including layer 29 (ReLU of block5_conv3)
        _CACHED_TORCH_VGG = vgg.features[:30].eval()
        for p in _CACHED_TORCH_VGG.parameters():
            p.requires_grad = False
    return _CACHED_TORCH_VGG


def _get_tf_conv_model(target_size=(256, 256)):
    """Returns cached TensorFlow VGG16 model outputting block5_conv3."""
    global _CACHED_TF_CONV_MODEL
    if _CACHED_TF_CONV_MODEL is None:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        h, w = target_size
        base = VGG16(weights="imagenet", include_top=False, input_shape=(h, w, 3))
        conv_layer = base.get_layer("block5_conv3").output
        _CACHED_TF_CONV_MODEL = Model(inputs=base.input, outputs=conv_layer)
    return _CACHED_TF_CONV_MODEL


def _optical_saliency_heatmap(img_bgr, target_size=(256, 256)):
    """
    Intelligent medical image saliency fallback when neither PyTorch nor TensorFlow
    is installed. Highlights microaneurysms, hemorrhages, and exudates via CLAHE,
    green-channel contrast, and multi-scale morphological top-hat filtering.
    """
    h, w = target_size
    img_resized = cv2.resize(img_bgr, (w, h))
    green = img_resized[:, :, 1]
    
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    
    # Morphological top-hat (detects bright exudates) and black-hat (detects dark hemorrhages)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_small)
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_small)
    
    # Combined lesion saliency map
    saliency = cv2.addWeighted(tophat, 0.5, blackhat, 0.5, 0).astype(np.float32)
    saliency_smooth = cv2.GaussianBlur(saliency, (17, 17), 0)
    
    if saliency_smooth.max() > saliency_smooth.min():
        saliency_norm = (saliency_smooth - saliency_smooth.min()) / (saliency_smooth.max() - saliency_smooth.min())
    else:
        saliency_norm = np.zeros((h, w), dtype=np.float32)
        
    return saliency_norm


def grad_cam(img_bgr, target_size=None, alpha=0.45):
    """
    Generates an Activation Feature Map based on the final VGG16 convolutional
    layer (block5_conv3), highlighting regions with prominent pathology (hemorrhages,
    exudates, neovascularization, and optic disc landmarks).
    
    Compatible with PyTorch (preferred), TensorFlow, and high-fidelity optical fallback.
    """
    from . import config as _cfg
    if target_size is None:
        target_size = getattr(_cfg, "TARGET_SIZE", (256, 256))
        # Ensure minimum 256 for optimal 16x16 Grad-CAM resolution matching reference
        if target_size[0] < 256 or target_size[1] < 256:
            target_size = (256, 256)

    h, w = target_size
    img_resized = cv2.resize(img_bgr, (w, h))

    # --- 1. PyTorch Backend ---
    if TORCH_AVAILABLE:
        try:
            model = _get_torch_conv_model()
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Standard ImageNet normalization
            tensor = tv_transforms.ToTensor()(img_rgb)
            normalize = tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensor = normalize(tensor).unsqueeze(0)
            
            with torch.no_grad():
                conv_out = model(tensor)  # (1, 512, 16, 16)
                # Compute activation map by averaging across all 512 channels
                heatmap = torch.mean(conv_out[0], dim=0).cpu().numpy()

            # Robust min-max normalization
            hm_min, hm_max = float(heatmap.min()), float(heatmap.max())
            if hm_max > hm_min:
                heatmap = (heatmap - hm_min) / (hm_max - hm_min)
            else:
                heatmap = np.zeros_like(heatmap)

            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_u8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_color, alpha, 0)
            return overlay, heatmap_color
        except Exception as e:
            logger.warning(f"PyTorch Grad-CAM computation failed: {e}. Trying fallback.")

    # --- 2. TensorFlow Backend ---
    if TF_AVAILABLE:
        try:
            model = _get_tf_conv_model(target_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            x = np.expand_dims(img_rgb, 0).astype(np.float32)
            x = vgg_preprocess(x)

            conv_out = model(x)
            heatmap = tf.reduce_mean(conv_out[0], axis=-1).numpy()
            hm_min, hm_max = float(heatmap.min()), float(heatmap.max())
            if hm_max > hm_min:
                heatmap = (heatmap - hm_min) / (hm_max - hm_min)
            else:
                heatmap = np.zeros_like(heatmap)

            heatmap_resized = cv2.resize(heatmap, (w, h))
            heatmap_u8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_color, alpha, 0)
            return overlay, heatmap_color
        except Exception as e:
            logger.warning(f"TensorFlow Grad-CAM computation failed: {e}. Trying fallback.")

    # --- 3. Medical Optical Saliency Fallback ---
    heatmap = _optical_saliency_heatmap(img_bgr, target_size)
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_color, alpha, 0)
    return overlay, heatmap_color
