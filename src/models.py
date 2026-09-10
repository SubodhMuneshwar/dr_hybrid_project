import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from . import config

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    # save scaler with model? (not needed since we scale before model pipelines, but keep if desired)
    return X_train, X_test, y_train, y_test

def get_base_pipelines():
    # Light-ish models + SMOTE inside pipeline to address imbalance during CV
    rf = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42))
    ])
    rf_params = {"rf__n_estimators": [100], "rf__max_depth": [None, 20]}

    svm = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("svm", SVC(probability=True, kernel="rbf", random_state=42))
    ])
    # keep SVM small to avoid Windows spawn slowness
    svm_params = {"svm__C": [1.0], "svm__gamma": ["scale"]}

    knn = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=42)),
        ("knn", KNeighborsClassifier())
    ])
    knn_params = {"knn__n_neighbors": [5, 7], "knn__weights": ["distance"]}

    return (rf, rf_params), (svm, svm_params), (knn, knn_params)

def tune_model(pipe, param_grid, X, y, cv=3):
    # Single-thread on Windows to avoid joblib spawn headaches
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=1, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_

def build_stacking(rf_best, svm_best, knn_best):
    # Simpler & faster: no CalibratedClassifierCV wrapping here
    estimators = [
        ("rf", rf_best),
        ("svm", svm_best),
        ("knn", knn_best)
    ]
    final_est = LogisticRegression(max_iter=200, n_jobs=None)
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_est,
        stack_method="predict_proba",
        n_jobs=1
    )
    return stack

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
