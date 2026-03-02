import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Adjust the import path based on your structure
from starter.ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    """
    Test that train_model returns the correct object type and is fitted.
    """
    # Create simple dummy data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    
    model = train_model(X, y)
    
    # Check if the returned object is the right class
    assert isinstance(model, RandomForestClassifier)
    # Check if the model has been fitted (fitted models have attributes ending in _)
    assert hasattr(model, "estimators_")

def test_compute_model_metrics():
    """
    Test the metrics calculation with known inputs.
    """
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 0, 1, 1])
    
    # Manually calculating for these arrays:
    # TP: 2, FP: 1, FN: 1, TN: 1
    # Precision: 2 / (2+1) = 0.666...
    # Recall: 2 / (2+1) = 0.666...
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert precision == pytest.approx(0.6666, abs=1e-2)
    assert recall == pytest.approx(0.6666, abs=1e-2)

def test_inference():
    """
    Test that inference returns the correct shape and binary values.
    """
    # Setup a fitted model
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model = train_model(X_train, y_train)
    
    # Data to predict
    X_test = np.array([[1, 2], [5, 6], [3, 4]])
    preds = inference(model, X_test)
    
    # Check shape
    assert len(preds) == 3
    # Check that output only contains 0 or 1
    assert np.all((preds == 0) | (preds == 1))
    # Check return type
    assert isinstance(preds, np.ndarray)