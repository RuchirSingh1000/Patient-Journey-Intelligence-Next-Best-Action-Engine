"""
Unit tests for the API module.
"""
import os
import sys
import tempfile
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.api.main import app
    from fastapi.testclient import TestClient
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

@pytest.mark.skipif(not API_AVAILABLE, reason="API not available")
class TestAPI:
    """Test cases for the API module."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "PharmaPulse API" in data["message"]

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "models_loaded" in data

    @patch('src.api.main.next_best_action_engine')
    def test_recommend_endpoint_no_engine(self, mock_engine):
        """Test recommend endpoint when engine is not available."""
        mock_engine = None

        response = self.client.post(
            "/recommend",
            json={"patient_id": "P00000"}
        )
        # Should return 503 when models not available
        assert response.status_code == 503

    @patch('src.api.main.next_best_action_engine')
    def test_recommend_endpoint_success(self, mock_engine):
        """Test recommend endpoint with successful response."""
        # Mock the engine
        mock_engine_instance = Mock()
        mock_engine_instance.recommend_next_best_action.return_value = {
            'patient_id': 'P00000',
            'timestamp': '2023-01-01T00:00:00',
            'patient_info': {
                'patient_id': 'P00000',
                'age': 45,
                'gender': 'Male',
                'ethnicity': 'Caucasian',
                'cluster': 0,
                'current_treatment': 1
            },
            'risk_scores': {
                'adherence_risk': 0.3,
                'discontinuation_risk': 0.1
            },
            'recommended_action': 'Continue current treatment',
            'action_details': 'Patient looks good',
            'supporting_evidence': {
                'adherence_risks': {},
                'discontinuation_risks': {},
                'uplift_treatment_a': {},
                'uplift_treatment_b': {},
                'clinical_guidelines': []
            }
        }
        mock_engine = mock_engine_instance

        with patch('src.api.main.next_best_action_engine', mock_engine):
            response = self.client.post(
                "/recommend",
                json={"patient_id": "P00000"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data['patient_id'] == 'P00000'
            assert data['recommended_action'] == 'Continue current treatment'

    @patch('src.api.main.next_best_action_engine')
    def test_recommend_endpoint_patient_not_found(self, mock_engine):
        """Test recommend endpoint with non-existent patient."""
        # Mock the engine to raise ValueError
        mock_engine_instance = Mock()
        mock_engine_instance.recommend_next_best_action.side_effect = ValueError("Patient ID P00999 not found in data.")
        mock_engine = mock_engine_instance

        with patch('src.api.main.next_best_action_engine', mock_engine):
            response = self.client.post(
                "/recommend",
                json={"patient_id": "P00999"}
            )
            assert response.status_code == 404

    @patch('src.api.main.evidence_retriever')
    def test_evidence_retrieve_endpoint(self, mock_retriever):
        """Test evidence retrieval endpoint."""
        # Mock the retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve_evidence.return_value = [
            {
                'id': 'TEST_001',
                'title': 'Test Guideline',
                'content': 'Test content',
                'condition': 'test',
                'category': 'test',
                'source': 'Test Org',
                'similarity_score': 0.85,
                'rank': 1
            }
        ]
        mock_retriever = mock_retriever_instance

        with patch('src.api.main.evidence_retriever', mock_retriever):
            response = self.client.post(
                "/evidence/retrieve",
                json={"query": "test query", "top_k": 1}
            )
            assert response.status_code == 200
            data = response.json()
            assert data['query'] == "test query"
            assert data['total_results'] == 1
            assert len(data['results']) == 1
            assert data['results'][0]['title'] == 'Test Guideline'

    @patch('src.api.main.evidence_retriever')
    def test_evidence_for_patient_endpoint(self, mock_retriever):
        """Test evidence for patient endpoint."""
        # Mock the retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve_evidence_for_patient.return_value = [
            {
                'id': 'TEST_001',
                'title': 'Test Guideline',
                'content': 'Test content',
                'condition': 'test',
                'category': 'test',
                'source': 'Test Org',
                'similarity_score': 0.85,
                'rank': 1
            }
        ]
        mock_retriever = mock_retriever_instance

        with patch('src.api.main.evidence_retriever', mock_retriever):
            response = self.client.post(
                "/evidence/patient",
                json={
                    "patient_id": "P00000",
                    "age": 45,
                    "gender": "Male",
                    "conditions": ["hypertension"]
                }
            )
            assert response.status_code == 200
            data = response.json()
            assert "Patient P00000" in data['query']
            assert data['total_results'] == 1
            assert len(data['results']) == 1

    @patch('src.api.main.patient_data')
    def test_stats_endpoint(self, mock_patient_data):
        """Test stats endpoint."""
        # Mock patient data
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_df.__getitem__ = Mock(side_effect=lambda x: Mock(
            mean=Mock(return_value=50.0),
            std=Mock(return_value=10.0),
            min=Mock(return_value=20.0),
            max=Mock(return_value=80.0),
            value_counts=Mock(return_value={'Male': 60, 'Female': 40})
        ) if x in ['age', 'gender'] else Mock(value_counts=Mock(return_value={})))
        mock_df.columns = ['age', 'gender', 'treatment', 'cluster']

        with patch('src.api.main.patient_data', mock_df):
            response = self.client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert data['total_patients'] == 100
            assert 'age_stats' in data
            assert 'gender_distribution' in data

    @patch('src.api.main.patient_data')
    def test_get_patient_endpoint(self, mock_patient_data):
        """Test get specific patient endpoint."""
        # Mock patient data for a specific patient
        mock_row = Mock()
        mock_row.__getitem__ = Mock(side_effect=lambda x: {
            'patient_id': 'P00000',
            'age': 45,
            'gender': 'Male',
            'ethnicity': 'Caucasian',
            'treatment': 1
        }.get(x, 'default'))
        mock_row.__len__ = Mock(return_value=1)

        mock_df = Mock()
        mock_df.__getitem__ = Mock(return_value=mock_row)
        mock_df.__len__ = Mock(return_value=1)

        with patch('src.api.main.patient_data', mock_df):
            response = self.client.get("/patients/P00000")
            assert response.status_code == 200
            data = response.json()
            assert data['patient_id'] == 'P00000'
            assert data['age'] == 45
            assert data['gender'] == 'Male'

    @patch('src.api.main.patient_data')
    def test_get_patient_endpoint_not_found(self, mock_patient_data):
        """Test get specific patient endpoint with non-existent patient."""
        # Mock patient data returning empty results
        mock_df = Mock()
        mock_df.__getitem__ = Mock(return_value=Mock(__len__=Mock(return_value=0)))

        with patch('src.api.main.patient_data', mock_df):
            response = self.client.get("/patients/P00999")
            assert response.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__])