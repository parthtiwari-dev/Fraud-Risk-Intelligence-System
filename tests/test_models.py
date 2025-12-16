import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to python path so we can import models
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import the module to be tested
import models

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_file_system(tmp_path):
    """Creates a temporary directory structure mimicking the real one."""
    
    # Define paths based on models.py structure
    base = tmp_path
    models_dir = base / "experiments" / "models"
    ensemble_dir = base / "experiments" / "ensemble"
    models_dir.mkdir(parents=True)
    ensemble_dir.mkdir(parents=True)

    # 1. Mock Feature Lists
    features_list = ["feat_1", "feat_2", "amount_log", "merchant_freq", "account_txn_count", "last_5_mean_amount", "cluster_id"]
    
    feature_file_content = json.dumps({"features": features_list})
    
    (models_dir / "xgb_features.json").write_text(feature_file_content)
    (models_dir / "iforest_features.json").write_text(feature_file_content)
    (models_dir / "ae_features.json").write_text(feature_file_content)

    # 2. Mock Metrics
    metrics_content = json.dumps({
        "meta_features": ["xgb_oof_proba", "anomaly_score", "ae_recon_error", "cluster_id", "amount_log", "merchant_freq", "account_txn_count", "last_5_mean_amount"],
        "final_threshold": 0.5
    })
    (ensemble_dir / "metrics.json").write_text(metrics_content)

    return base

@pytest.fixture
def mock_models():
    """Mocks the actual ML models."""
    
    # Mock Scikit-Learn/XGBoost models
    mock_xgb = MagicMock()
    # predict_proba returns [prob_class_0, prob_class_1]
    mock_xgb.predict_proba.return_value = np.array([[0.1, 0.9]]) 
    
    mock_iforest = MagicMock()
    # decision_function returns scores
    mock_iforest.decision_function.return_value = np.array([-0.5]) 
    
    mock_stacker = MagicMock()
    mock_stacker.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    # Mock PyTorch State Dict
    # Create a real small autoencoder to get a valid state_dict
    real_ae = models.Autoencoder(input_dim=7) # 7 features in our mock list
    state_dict = real_ae.state_dict()

    return {
        "xgb": mock_xgb,
        "iforest": mock_iforest,
        "stacker": mock_stacker,
        "ae_state": state_dict
    }

# --- Tests ---

def test_load_models_structure(mock_file_system, mock_models):
    """Test if load_models caches correctly and loads all artifacts."""
    
    # Patch the paths in models.py to point to our temp dir
    with patch("models.BASE_DIR", mock_file_system), \
         patch("models.METRICS_PATH", mock_file_system / "experiments" / "ensemble" / "metrics.json"), \
         patch("models.XGB_FEATURES_PATH", mock_file_system / "experiments" / "models" / "xgb_features.json"), \
         patch("models.IFOREST_FEATURES_PATH", mock_file_system / "experiments" / "models" / "iforest_features.json"), \
         patch("models.AE_FEATURES_PATH", mock_file_system / "experiments" / "models" / "ae_features.json"), \
         patch("models.load", side_effect=[mock_models["stacker"], mock_models["xgb"], mock_models["iforest"]]), \
         patch("torch.load", return_value=mock_models["ae_state"]):

        # Reset global cache for test
        models._MODELS = None
        
        loaded_artifacts = models.load_models()
        
        assert "stacker" in loaded_artifacts
        assert "xgb" in loaded_artifacts
        assert "autoencoder" in loaded_artifacts
        assert loaded_artifacts["threshold"] == 0.5
        assert isinstance(loaded_artifacts["autoencoder"], nn.Module)

def test_prediction_flow(mock_file_system, mock_models):
    """Test the full end-to-end prediction flow."""
    
    # Input data matching our mock feature list
    input_data = {
        "feat_1": 1.0,
        "feat_2": 0.5,
        "amount_log": 2.3,
        "merchant_freq": 5,
        "account_txn_count": 10,
        "last_5_mean_amount": 50.0,
        "cluster_id": 1
    }

    # Apply same patches as above
    with patch("models.BASE_DIR", mock_file_system), \
         patch("models.METRICS_PATH", mock_file_system / "experiments" / "ensemble" / "metrics.json"), \
         patch("models.XGB_FEATURES_PATH", mock_file_system / "experiments" / "models" / "xgb_features.json"), \
         patch("models.IFOREST_FEATURES_PATH", mock_file_system / "experiments" / "models" / "iforest_features.json"), \
         patch("models.AE_FEATURES_PATH", mock_file_system / "experiments" / "models" / "ae_features.json"), \
         patch("models.load", side_effect=[mock_models["stacker"], mock_models["xgb"], mock_models["iforest"]]), \
         patch("torch.load", return_value=mock_models["ae_state"]):

        models._MODELS = None
        
        # Test predict_proba
        result = models.predict(input_data)
        
        # Check structure
        assert "score" in result
        assert "label" in result
        assert result["label"] in ["fraud", "legit"]
        
        # Since we mocked stacker to return 0.8 and threshold is 0.5
        assert result["score"] == 0.8
        assert result["label"] == "fraud"

def test_missing_features_error(mock_file_system, mock_models):
    """Ensure code raises error when input features are missing."""
    
    incomplete_data = {"feat_1": 1.0} # Missing many features

    with patch("models.BASE_DIR", mock_file_system), \
         patch("models.METRICS_PATH", mock_file_system / "experiments" / "ensemble" / "metrics.json"), \
         patch("models.XGB_FEATURES_PATH", mock_file_system / "experiments" / "models" / "xgb_features.json"), \
         patch("models.IFOREST_FEATURES_PATH", mock_file_system / "experiments" / "models" / "iforest_features.json"), \
         patch("models.AE_FEATURES_PATH", mock_file_system / "experiments" / "models" / "ae_features.json"), \
         patch("models.load", side_effect=[mock_models["stacker"], mock_models["xgb"], mock_models["iforest"]]), \
         patch("torch.load", return_value=mock_models["ae_state"]):

        models._MODELS = None
        
        with pytest.raises(ValueError) as excinfo:
            models.predict(incomplete_data)
        
        assert "Missing required features" in str(excinfo.value)
