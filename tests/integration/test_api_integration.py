"""
Integration tests for the API module.
These tests actually start the API and test endpoints in a more realistic scenario.
"""
import os
import sys
import tempfile
import json
import pytest
import time
import threading
import requests
from unittest.mock import Mock, patch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try to import necessary modules
try:
    import uvicorn
    from src.api.main import app
    INTEGRATION_TEST_POSSIBLE = True
except ImportError:
    INTEGRATION_TEST_POSSIBLE = False

@pytest.mark.skipif(not INTEGRATION_TEST_POSSIBLE, reason="Integration test dependencies not available")
class TestAPIIntegration:
    """Integration test cases for the API module."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures before running tests."""
        cls.server_thread = None
        cls.base_url = "http://localhost:8001"  # Use different port to avoid conflicts
        cls.server_started = False

    @classmethod
    def teardown_class(cls):
        """Clean up after running tests."""
        # Note: In a real test, we would shut down the server here
        # For simplicity, we'll just note that tests completed
        pass

    def start_test_server(self):
        """Start the test API server in a background thread."""
        def run_server():
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=8001,
                log_level="error"  # Reduce logging during tests
            )

        cls.server_thread = threading.Thread(target=run_server, daemon=True)
        cls.server_thread.start()
        # Wait for server to start
        time.sleep(3)
        cls.server_started = True

    def test_server_starts_and_responds(self):
        """Test that the server starts and responds to basic requests."""
        if not self.server_started:
            self.start_test_server()

        # Give it a moment to be ready
        time.sleep(1)

        try:
            # Test health endpoint
            response = requests.get(f"{self.base_url}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data

            # Test root endpoint
            response = requests.get(f"{self.base_url}/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "PharmaPulse API" in data["message"]

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Server request failed: {e}")

    @patch('src.api.main.next_best_action_engine')
    def test_recommend_endpoint_integration(self, mock_engine):
        """Test the recommend endpoint with mocked engine."""
        if not self.server_started:
            self.start_test_server()
            time.sleep(1)

        # Mock the engine to return a specific response
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
                'adherence_risk': 0.25,
                'discontinuation_risk': 0.05
            },
            'recommended_action': 'Continue current treatment with routine monitoring',
            'action_details': 'Risks are low, continue current plan',
            'supporting_evidence': {
                'adherence_risks': {'logistic_regression_adherence': {
                    'prob_high_adherence': 0.75,
                    'adherence_risk': 0.25
                }},
                'discontinuation_risks': {},
                'uplift_treatment_a': {},
                'uplift_treatment_b': {},
                'clinical_guidelines': []
            }
        }
        mock_engine = mock_engine_instance

        with patch('src.api.main.next_best_action_engine', mock_engine):
            try:
                response = requests.post(
                    f"{self.base_url}/recommend",
                    json={"patient_id": "P00000"},
                    timeout=5
                )
                assert response.status_code == 200
                data = response.json()
                assert data['patient_id'] == 'P00000'
                assert data['recommended_action'] == 'Continue current treatment with routine monitoring'
                assert data['risk_scores']['adherence_risk'] == 0.25

            except requests.exceptions.RequestException as e:
                pytest.fail(f"Recommendation request failed: {e}")

    @patch('src.api.main.evidence_retriever')
    def test_evidence_endpoint_integration(self, mock_retriever):
        """Test the evidence retrieval endpoint with mocked retriever."""
        if not self.server_started:
            self.start_test_server()
            time.sleep(1)

        # Mock the retriever
        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve_evidence.return_value = [
            {
                'id': 'HTN_001',
                'title': 'Hypertension Management Guidelines',
                'content': 'For patients with hypertension, first-line treatment includes ACE inhibitors...',
                'condition': 'hypertension',
                'category': 'treatment',
                'source': 'American Heart Association',
                'similarity_score': 0.82,
                'rank': 1
            }
        ]
        mock_retriever = mock_retriever_instance

        with patch('src.api.main.evidence_retriever', mock_retriever):
            try:
                response = requests.post(
                    f"{self.base_url}/evidence/retrieve",
                    json={"query": "hypertension patient", "top_k": 1},
                    timeout=5
                )
                assert response.status_code == 200
                data = response.json()
                assert data['query'] == "hypertension patient"
                assert data['total_results'] == 1
                assert len(data['results']) == 1
                assert data['results'][0]['title'] == 'Hypertension Management Guidelines'
                assert data['results'][0]['similarity_score'] == 0.82

            except requests.exceptions.RequestException as e:
                pytest.fail(f"Evidence retrieval request failed: {e}")

    def test_stats_endpoint_integration(self):
        """Test the stats endpoint."""
        if not self.server_started:
            self.start_test_server()
            time.sleep(1)

        try:
            response = requests.get(f"{self.base_url}/stats", timeout=5)
            # This might fail if models aren't loaded, which is okay for this test
            # We're mainly checking that the endpoint responds
            assert response.status_code in [200, 503]  # Either success or service unavailable

            if response.status_code == 200:
                data = response.json()
                assert 'total_patients' in data

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Stats request failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__ -v])