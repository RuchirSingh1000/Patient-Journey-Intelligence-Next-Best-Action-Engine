"""
Unit tests for the next-best-action engine module.
"""
import os
import sys
import tempfile
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.next_best_action import NextBestActionEngine
    NEXT_BEST_ACTION_AVAILABLE = True
except ImportError:
    NEXT_BEST_ACTION_AVAILABLE = False

@pytest.mark.skipif(not NEXT_BEST_ACTION_AVAILABLE, reason="Next-best-action engine not available")
class TestNextBestActionEngine:
    """Test cases for NextBestActionEngine class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.processed_data_dir = os.path.join(self.data_dir, "processed_data")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Create sample patient features data
        self.sample_features = pd.DataFrame({
            'patient_id': ['P00000', 'P00001', 'P00002'],
            'age': [45, 65, 35],
            'gender': ['Male', 'Female', 'Male'],
            'ethnicity': ['Caucasian', 'African American', 'Hispanic'],
            'treatment': [1, 2, 1],
            'time_to_discontinuation': [300, 150, 400],
            'discontinued': [0, 1, 0],
            'cluster': [0, 1, 0],
            'pca_1': [0.5, -0.3, 0.8],
            'pca_2': [-0.2, 0.6, 0.1],
            # Add some feature columns that models might expect
            'feature_1': [1.0, 2.0, 1.5],
            'feature_2': [0.5, 1.5, 0.8],
            'adherence_score': [0.8, 0.3, 0.9]
        })

        # Create sample patient segments data
        self.sample_segments = pd.DataFrame({
            'patient_id': ['P00000', 'P00001', 'P00002'],
            'cluster': [0, 1, 0],
            'pca_1': [0.5, -0.3, 0.8],
            'pca_2': [-0.2, 0.6, 0.1]
        })

        # Save sample data
        self.sample_features.to_csv(os.path.join(self.processed_data_dir, "patient_features.csv"), index=False)
        self.sample_segments.to_csv(os.path.join(self.processed_data_dir, "patient_segments.csv"), index=False)

        # Create mock model files
        self._create_mock_models()

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_models(self):
        """Create mock model files for testing."""
        import pickle
        from sklearn.linear_model import LogisticRegression

        # Create predictive models directory
        pred_dir = os.path.join(self.models_dir, "predictive")
        os.makedirs(pred_dir, exist_ok=True)

        # Create a simple mock model
        mock_model = LogisticRegression()
        # Fit with dummy data to make it usable
        X_dummy = np.array([[1, 2], [2, 3], [3, 4]])
        y_dummy = np.array([0, 1, 0])
        mock_model.fit(X_dummy, y_dummy)

        # Save mock model
        model_path = os.path.join(pred_dir, "logistic_regression_adherence_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)

        # Save mock preprocessors
        preprocessors = {
            'label_encoders': {
                'gender': Mock(),
                'ethnicity': Mock()
            },
            'scaler': Mock(),
            'feature_names': ['feature_1', 'feature_2', 'adherence_score']
        }
        # Mock the label encoders
        preprocessors['label_encoders']['gender'].classes_ = ['Female', 'Male']
        preprocessors['label_encoders']['gender'].transform = lambda x: [0 if i == 'Female' else 1 for i in x]
        preprocessors['label_encoders']['ethnicity'].classes_ = ['African American', 'Caucasian', 'Hispanic']
        preprocessors['label_encoders']['ethnicity'].transform = lambda x: [0 if i == 'African American' else 1 if i == 'Caucasian' else 2 for i in x]
        # Mock the scaler
        preprocessors['scaler'].transform = lambda x: x  # Identity transform

        prep_path = os.path.join(pred_dir, "adherence_preprocessors.pkl")
        with open(prep_path, 'wb') as f:
            pickle.dump(preprocessors, f)

        # Create survival models directory
        surv_dir = os.path.join(self.models_dir, "survival")
        os.makedirs(surv_dir, exist_ok=True)

        # Create a mock KaplanMeierFitter
        from lifelines import KaplanMeierFitter
        mock_kmf = KaplanMeierFitter()
        # Fit with dummy data
        mock_kmf.fit([100, 200, 300], [1, 0, 1])

        kmf_path = os.path.join(surv_dir, "kmf_overall.pkl")
        with open(kmf_path, 'wb') as f:
            pickle.dump(mock_kmf, f)

        # Create causal models directory
        causal_dir = os.path.join(self.models_dir, "causal")
        os.makedirs(causal_dir, exist_ok=True)

        # Create a mock uplift model (simple dictionary structure)
        mock_uplift_model = {
            'treatment': mock_model,
            'control': mock_model
        }
        uplift_path = os.path.join(causal_dir, "mock_uplift_model_model.pkl")
        with open(uplift_path, 'wb') as f:
            pickle.dump(mock_uplift_model, f)

        # Mock preprocessors for uplift model
        uplift_preprocessors = {
            'label_encoders': {
                'gender': Mock(),
                'ethnicity': Mock()
            },
            'scaler': Mock(),
            'feature_names': ['feature_1', 'feature_2', 'adherence_score']
        }
        # Mock the label encoders (same as above)
        uplift_preprocessors['label_encoders']['gender'].classes_ = ['Female', 'Male']
        uplift_preprocessors['label_encoders']['gender'].transform = lambda x: [0 if i == 'Female' else 1 for i in x]
        uplift_preprocessors['label_encoders']['ethnicity'].classes_ = ['African American', 'Caucasian', 'Hispanic']
        uplift_preprocessors['label_encoders']['ethnicity'].transform = lambda x: [0 if i == 'African American' else 1 if i == 'Caucasian' else 2 for i in x]
        # Mock the scaler
        uplift_preprocessors['scaler'].transform = lambda x: x  # Identity transform

        uplift_prep_path = os.path.join(causal_dir, "treatment 1_vs_control_adherence_preprocessors.pkl")
        with open(uplift_prep_path, 'wb') as f:
            pickle.dump(uplift_preprocessors, f)

    @patch('src.models.next_best_action.EVIDENCE_RETRIEVER_AVAILABLE', False)
    def test_initialization_without_evidence_retriever(self):
        """Test initialization when evidence retriever is not available."""
        engine = NextBestActionEngine(
            models_dir=self.models_dir,
            data_dir=self.processed_data_dir
        )

        assert engine is not None
        assert engine.models_dir == self.models_dir
        assert engine.data_dir == self.processed_data_dir
        assert engine.evidence_retriever is None
        assert len(engine.models) > 0  # Should have loaded some models

    def test_load_patient_data(self):
        """Test loading patient data."""
        engine = NextBestActionEngine(
            models_dir=self.models_dir,
            data_dir=self.processed_data_dir
        )

        assert engine.feature_data is not None
        assert engine.segment_data is not None
        assert engine.patient_data is not None
        assert len(engine.patient_data) == 3  # Should have 3 patients
        assert 'patient_id' in engine.patient_data.columns

    def test_prepare_patient_features(self):
        """Test preparing patient features for a model."""
        engine = NextBestActionEngine(
            models_dir=self.models_dir,
            data_dir=self.processed_data_dir
        )

        # Test with first patient
        patient_id = "P00000"
        # Get the first model name
        model_names = list(engine.models.get('predictive', {}).keys())
        if model_names:
            model_name = model_names[0]
            features = engine._prepare_patient_features(patient_id, model_name)
            assert features is not None
            assert isinstance(features, np.ndarray)
            assert features.shape[0] == 1  # One patient
            assert features.shape[1] > 0   # Some features

    def test_predict_adherence_risk(self):
        """Test predicting adherence risk."""
        engine = NextBestActionEngine(
            models_dir=self.models_dir,
            data_dir=self.processed_data_dir
        )

        patient_id = "P00000"
        risks = engine.predict_adherence_risk(patient_id)
        # Should return a dictionary (might be empty if no adherence models)
        assert isinstance(risks, dict)

    def test_predict_discontinuation_risk(self):
        """Test predicting discontinuation risk."""
        engine = NextBestActionEngine(
            models_dir=self.models_dir,
            data_dir=self.processed_data_dir
        )

        patient_id = "P00000"
        risks = engine.predict_discontinuation_risk(patient_id)
        # Should return a dictionary (might be empty if no discontinuation models)
        assert isinstance(risks, dict)

    def test_recommend_next_best_action(self):
        """Test generating a next-best-action recommendation."""
        engine = NextBestActionEngine(
            models_dir=self.models_dir,
            data_dir=self.processed_data_dir
        )

        patient_id = "P00000"
        recommendation = engine.recommend_next_best_action(patient_id)

        # Check structure of recommendation
        assert isinstance(recommendation, dict)
        assert 'patient_id' in recommendation
        assert 'timestamp' in recommendation
        assert 'patient_info' in recommendation
        assert 'risk_scores' in recommendation
        assert 'recommended_action' in recommendation
        assert 'action_details' in recommendation
        assert 'supporting_evidence' in recommendation

        # Check data types
        assert recommendation['patient_id'] == patient_id
        assert isinstance(recommendation['risk_scores'], dict)
        assert isinstance(recommendation['recommended_action'], str)
        assert isinstance(recommendation['action_details'], str)

    def test_get_population_recommendations(self):
        """Test generating recommendations for population."""
        engine = NextBestActionEngine(
            models_dir=self.models_dir,
            data_dir=self.processed_data_dir
        )

        recommendations = engine.get_population_recommendations()
        assert isinstance(recommendations, list)
        # Should have recommendations for each patient
        assert len(recommendations) <= len(engine.patient_data)

        # Check structure of first recommendation if any exist
        if recommendations:
            rec = recommendations[0]
            assert isinstance(rec, dict)
            assert 'patient_id' in rec
            assert 'recommended_action' in rec

if __name__ == "__main__":
    pytest.main([__file__])