import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.models import Model

def _vgg16_conv_model():
    """Builds a model that outputs the final convolution block of VGG16"""
    base = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    # 'block5_pool' is what the feature extractor uses, but we want the conv before pooling
    # for spatial resolution. We will take 'block5_conv3'
    conv_layer = base.get_layer("block5_conv3").output
    model = Model(inputs=base.input, outputs=conv_layer)
    return model

def grad_cam(img_bgr, target_size=(256, 256), alpha=0.45):
    """
    Since the top model is an sklearn Voting Classifier (not a neural network), 
    we cannot compute standard Grad-CAM gradients back to the image. 
    
    Instead, this generates an Activation Feature Map based on the L2 norm 
    of the final VGG16 convolutional layer, visualizing where the deep 
    extractor found the strongest structural anomalies (like hemorrhages).
    """
    model = _vgg16_conv_model()
    
    # Preprocess exactly matching VGG16
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, target_size), cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img_rgb, 0).astype(np.float32)
    x = vgg_preprocess(x)

    # Get the raw convolutional feature maps (Shape: 1, 16, 16, 512)
    conv_out = model(x)
    
    # Compute the activation map by taking the spatial mean across all 512 filters
    heatmap = tf.reduce_mean(conv_out[0], axis=-1).numpy()
    
    # Normalize between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
        
    # Resize back to original image target size
    heatmap = cv2.resize(heatmap, (target_size[0], target_size[1]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply JET colormap for visualization
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay onto original image
    overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay, heatmap_color
