"""
MODEL IMPROVEMENT STRATEGIES
Detailed guide to improve your Stacking Classifier
"""

IMPROVEMENT_GUIDE = """

╔════════════════════════════════════════════════════════════════════════════════╗
║                     MODEL IMPROVEMENT STRATEGIES                              ║
╚════════════════════════════════════════════════════════════════════════════════╝

YOUR CURRENT MODEL:
  - Architecture: Stacking Classifier (RF + SVM + KNN, combined with Logistic Regression)
  - Features: DenseNet121 deep features + LBP + Haralick
  - Training: GridSearchCV with SMOTE for class imbalance
  - Problem: Low confidence on harder classes (Class 1, 3)

═══════════════════════════════════════════════════════════════════════════════════

OPTION 1: HYPERPARAMETER TUNING (Easiest - No new training data needed)
═════════════════════════════════════════════════════════════════════════════════

What it does:
  - Fine-tune existing models to find better parameter settings
  - No retraining from scratch, just optimization

Steps:
  1. Open: src/models.py (lines with param_grid definitions)
  2. Expand the search space:

    BEFORE (line ~38):
      rf_params = {"rf__n_estimators": [100], "rf__max_depth": [None, 20]}
    
    AFTER (Better tuning):
      rf_params = {
          "rf__n_estimators": [100, 200, 300],
          "rf__max_depth": [15, 20, 25, 30],
          "rf__min_samples_split": [2, 5, 10]
      }

  3. For SVM (currently too conservative):
      svm_params = {
          "svm__C": [0.1, 1.0, 10.0, 100.0],
          "svm__gamma": ["scale", "auto", 0.001, 0.01],
          "svm__kernel": ["rbf", "poly"]  # Add polynomial kernel
      }

  4. For KNN:
      knn_params = {
          "knn__n_neighbors": [3, 5, 7, 9, 11],
          "knn__weights": ["uniform", "distance"],
          "knn__metric": ["euclidean", "manhattan"]
      }

  5. Run: python -m src.pipeline train
     (This will retrain with better hyperparameters)

Expected Improvement: +3-8% accuracy

⏱️  Time: ~2-4 hours (depends on CV folds)
💾 Resources: Moderate CPU usage
🔧 Difficulty: Easy


═══════════════════════════════════════════════════════════════════════════════════

OPTION 2: BETTER FEATURE ENGINEERING (Moderate - Some effort)
═════════════════════════════════════════════════════════════════════════════════

What it does:
  - Extract richer features that better capture DR patterns
  - Replace or augment current features

Current Features (in src/features.py):
  ✓ DenseNet121 (2048-dim) - Good for general patterns
  ✓ LBP (59-dim) - Good for texture
  ✓ Haralick (13-dim) - Good for texture variation
  Total: 2120 dimensions

Improvements:

A) USE STRONGER FEATURE EXTRACTOR:
   
   Current:    config.FEATURE_EXTRACTOR_MODEL = "densenet121"
   
   Change to:  config.FEATURE_EXTRACTOR_MODEL = "inceptionresnetv2"  (better accuracy)
               config.FEATURE_EXTRACTOR_MODEL = "mobilenetv2"        (faster, lighter)
   
   Expected gain: +2-4% accuracy
   Time: ~3-6 hours for feature extraction
   
B) ADD VESSEL DETECTION FEATURES:
   
   DR causes blood vessel changes. Add vessel-specific features:
   
   In src/features.py, add:
   ```python
   def extract_vessel_features(img_clahe):
       # Apply morphological operations to highlight vessels
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
       dilated = cv2.dilate(img_clahe, kernel, iterations=3)
       eroded = cv2.erode(dilated, kernel, iterations=3)
       
       # Count white regions (vessels)
       vessel_density = np.sum(eroded > 200) / (eroded.shape[0] * eroded.shape[1])
       vessel_variance = np.var(eroded)
       
       return np.array([vessel_density, vessel_variance])
   ```
   
   Expected gain: +1-3% accuracy
   
C) COLOR-BASED HEMORRHAGE FEATURES:
   
   Hemorrhages have specific red/orange colors. Add color features:
   
   ```python
   def extract_hemorrhage_features(img_bgr):
       # Convert to HSV for color-based detection
       hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
       
       # Red hemorrhages (specific hue range)
       lower_red = np.array([0, 50, 50])
       upper_red = np.array([10, 255, 255])
       mask_red = cv2.inRange(hsv, lower_red, upper_red)
       
       red_area = np.sum(mask_red > 0) / (mask_red.shape[0] * mask_red.shape[1])
       
       return np.array([red_area])
   ```
   
   Expected gain: +2-5% accuracy

Combined feature engineering: +5-12% potential improvement
Time: ~4-8 hours
Difficulty: Moderate


═══════════════════════════════════════════════════════════════════════════════════

OPTION 3: CLASS IMBALANCE HANDLING (Easy - Very effective)
═════════════════════════════════════════════════════════════════════════════════

Problem in your data:
  - Class 0 (No DR): 359 samples   ← Dominant
  - Class 1 (Mild):  74 samples    ← Scarce ❌
  - Class 2 (Moderate): 199 samples
  - Class 3 (Severe): 39 samples   ← Very scarce ❌
  - Class 4 (Proliferative): 59 samples

Current solution: SMOTE (in models.py line 15)

Better solutions (in order of effectiveness):

A) INCREASE SMOTE RATIO:
   
   Current (line 19):
     ("smote", SMOTE(random_state=42))
   
   Change to:
     ("smote", SMOTE(sampling_strategy=0.8, random_state=42))
   
   This samples minority classes to 80% of majority class.
   
B) USE CLASS WEIGHTS:
   
   In pipeline.py, change:
   
   From:
     final_est = LogisticRegression(max_iter=200)
   
   To:
     final_est = LogisticRegression(max_iter=200, class_weight='balanced')
   
   For RF also:
     "rf", RandomForestClassifier(class_weight='balanced', random_state=42)

C) COMBINE SMOTE WITH UNDERSAMPLING:
   
   ```python
   from imblearn.combine import SMOTEENN
   
   pipeline = ImbPipeline([
       ('smote_enn', SMOTEENN(random_state=42)),
       ('rf', RandomForestClassifier(...))
   ])
   ```

Expected improvement: +2-8% (especially for Class 1 & 3)
Time: 30 minutes
Difficulty: Very Easy


═══════════════════════════════════════════════════════════════════════════════════

OPTION 4: ENSEMBLE STACKING IMPROVEMENTS (Moderate)
═════════════════════════════════════════════════════════════════════════════════

Current base models: Random Forest, SVM, KNN

Better combination:

Option 4A - ADD MORE BASE MODELS:
   
   ```python
   from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
   
   estimators = [
       ("rf", rf_best),
       ("svm", svm_best),
       ("knn", knn_best),
       ("gb", GradientBoostingClassifier(n_estimators=100)),  # NEW
       ("et", ExtraTreesClassifier(n_estimators=100))         # NEW
   ]
   ```
   
   Expected gain: +2-4%

Option 4B - BETTER META-LEARNER:
   
   Current final estimator: LogisticRegression
   
   Try:
     - RandomForestClassifier (captures non-linear relationships)
     - GradientBoostingClassifier (learns from stacking errors)
   
   Expected gain: +1-3%

Option 4C - USE VOTING CLASSIFIER INSTEAD:
   
   Your voting model might actually be better:
   
   In src/infer.py, change:
     path = os.path.join(config.MODELS_DIR, "votingclassifier_model.pkl")
   
   Then test both models to see which is better.

Expected improvement: +3-8%
Time: 1-2 hours
Difficulty: Easy-Moderate


═══════════════════════════════════════════════════════════════════════════════════

OPTION 5: DATA AUGMENTATION (Moderate - If you have GPU)
═════════════════════════════════════════════════════════════════════════════════

What it does:
  - Generate synthetic training samples from existing images
  - Increases training diversity without collecting new data

Methods:

A) GEOMETRIC TRANSFORMATIONS:
   ```python
   import albumentations as A
   
   transform = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.VerticalFlip(p=0.5),
       A.Rotate(limit=30, p=0.5),
       A.RandomBrightnessContrast(p=0.3),
       A.GaussNoise(p=0.2),
   ])
   ```

B) NEW DATA GENERATION (if DenseNet features are cached):
   ```python
   # Rotate original images and extract features again
   # This creates more training data without collection
   ```

Expected improvement: +3-7%
Time: 1-3 hours
Difficulty: Moderate


═══════════════════════════════════════════════════════════════════════════════════

OPTION 6: DEEP TRANSFER LEARNING FINE-TUNING (Advanced)
═════════════════════════════════════════════════════════════════════════════════

What it does:
  - Fine-tune DenseNet121 weights on your DR dataset
  - Currently DenseNet is frozen (feature extractor only)

Steps:

A) Unfreeze lower layers of DenseNet:
   
   ```python
   def get_finetunable_model():
       base_model = keras.applications.DenseNet121(
           weights='imagenet', 
           include_top=False
       )
       # Unfreeze last 20 layers
       for layer in base_model.layers[:-20]:
           layer.trainable = False
       for layer in base_model.layers[-20:]:
           layer.trainable = True
       return base_model
   ```

B) Train with lower learning rate:
   ```python
   optimizer = keras.optimizers.Adam(learning_rate=1e-5)
   ```

Expected improvement: +5-15%
Time: 4-12 hours (with good GPU) or 24+ hours (CPU)
Difficulty: Advanced
Requirements: GPU recommended


═══════════════════════════════════════════════════════════════════════════════════

RECOMMENDED IMPROVEMENT ROADMAP (Quickest wins first):
═════════════════════════════════════════════════════════════════════════════════

WEEK 1 (Easiest gains):
  1. Class Imbalance: Add class_weight='balanced' (30 min)          → +2-4%
  2. Hyperparameter Tuning: Expand search space (2-4 hours)         → +3-8%
  Cumulative: +5-12% improvement ✅

WEEK 2 (Moderate effort):
  3. Feature Engineering: Add vessel/hemorrhage features (4-8 hrs)   → +5-12%
  4. Better Meta-learner in stacking (1-2 hours)                    → +1-3%
  Cumulative: +10-20% improvement ✅

WEEK 3+ (Advanced):
  5. Deep learning fine-tuning (4-20 hours)                         → +5-15%
  6. Data augmentation (1-3 hours)                                  → +3-7%

EXPECTED FINAL RESULT:
  Current: ~82.7% accuracy (from your report)
  After all improvements: ~85-95% accuracy ✅


═══════════════════════════════════════════════════════════════════════════════════

HOW TO MEASURE IMPROVEMENT:
═════════════════════════════════════════════════════════════════════════════════

1. Run TEST_MODEL.py before changes:
   python TEST_MODEL.py > baseline_results.txt

2. Make one change (e.g., class weights)

3. Retrain:
   python -m src.pipeline train

4. Test again:
   python TEST_MODEL.py > new_results.txt

5. Compare:
   - Overall accuracy (should increase)
   - Per-class accuracy (especially Class 1 & 3)
   - Model confidence scores (higher is better)
   - Confusion matrix (fewer off-diagonal entries)

If improvement > +2%: Keep the change ✅
If improvement < +1%: Revert and try something else ❌


═══════════════════════════════════════════════════════════════════════════════════

TECHNICAL TIPS:
═════════════════════════════════════════════════════════════════════════════════

1. ALWAYS use cross-validation to validate improvements
2. Change ONE thing at a time (scientific method)
3. Save trained models with version numbers
4. Track your improvements in a spreadsheet
5. Focus on low-performing classes first (Class 1, 3)
6. Monitor for overfitting (training accuracy >> test accuracy)


═══════════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(IMPROVEMENT_GUIDE)
    
    # Save to file
    with open("MODEL_IMPROVEMENT_GUIDE.txt", "w") as f:
        f.write(IMPROVEMENT_GUIDE)
    print("\n✅ Guide saved to MODEL_IMPROVEMENT_GUIDE.txt")
