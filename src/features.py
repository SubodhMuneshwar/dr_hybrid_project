import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2, DenseNet121, InceptionResNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_preprocess
    TF_AVAILABLE = True
except ImportError as e:
    tf = None
    TF_AVAILABLE = False
    # Fallback stubs to allow Flask UI to load without TF (inference will error helpfully)
    MobileNetV2 = DenseNet121 = InceptionResNetV2 = VGG16 = Model = None
    def vgg_preprocess(x): return x
    def mobilenet_preprocess(x): return x
    def densenet_preprocess(x): return x
    def inception_preprocess(x): return x
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def get_deep_feature_model(model_name):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Install with: pip install -r requirements.txt (requires Python 3.10-3.11 for TF support)")
    if model_name == "mobilenetv2":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        model = Model(base.input, x)
        return model, mobilenet_preprocess
    elif model_name == "densenet121":
        base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        model = Model(base.input, x)
        return model, densenet_preprocess
    elif model_name == "inceptionresnetv2":
        base = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        model = Model(base.input, x)
        return model, inception_preprocess
    else:
        # Use GAP for VGG16 as well to keep feature dim small (512) and consistent across extractors
        # Previous version used block5_pool flatten (32768-d) which caused 32k vector bloat and train/infer size mismatch
        base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        model = Model(base.input, x)
        return model, vgg_preprocess

def extract_deep_features(img_bgr, deep_model, preprocess_fn):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available – cannot extract deep features")
    arr = tf.keras.preprocessing.image.img_to_array(img_bgr)  # Keep BGR, do not flip, since original model trained without flip
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_fn(arr)
    feats = deep_model.predict(arr, verbose=0)
    return feats.flatten()

def extract_lbp(img_gray):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_haralick(img_gray):
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    feats = np.hstack([
        graycoprops(glcm, 'contrast').ravel(),
        graycoprops(glcm, 'dissimilarity').ravel(),
        graycoprops(glcm, 'homogeneity').ravel(),
        graycoprops(glcm, 'energy').ravel(),
        graycoprops(glcm, 'correlation').ravel(),
        graycoprops(glcm, 'ASM').ravel()
    ])
    return feats
