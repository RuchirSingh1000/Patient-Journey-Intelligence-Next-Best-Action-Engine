"""
Unit tests for the evidence retriever module.
"""
import os
import sys
import tempfile
import json
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.evidence_retrieval.evidence_retriever import ClinicalGuidelinesEvidenceRetriever
    EVIDENCE_RETRIEVER_AVAILABLE = True
except ImportError:
    EVIDENCE_RETRIEVER_AVAILABLE = False

@pytest.mark.skipif(not EVIDENCE_RETRIEVER_AVAILABLE, reason="Evidence retriever not available")
class TestClinicalGuidelinesEvidenceRetriever:
    """Test cases for ClinicalGuidelinesEvidenceRetriever class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.guidelines_dir = os.path.join(self.temp_dir, "guidelines")
        self.index_path = os.path.join(self.temp_dir, "index.faiss")
        self.metadata_path = os.path.join(self.temp_dir, "metadata.pkl")

        os.makedirs(self.guidelines_dir, exist_ok=True)

        # Create sample guidelines for testing
        self.sample_guidelines = [
            {
                "id": "TEST_001",
                "title": "Test Guideline 1",
                "content": "This is a test guideline about hypertension treatment.",
                "condition": "hypertension",
                "category": "treatment",
                "source": "Test Organization"
            },
            {
                "id": "TEST_002",
                "title": "Test Guideline 2",
                "content": "This is a test guideline about diabetes management.",
                "condition": "diabetes",
                "category": "treatment",
                "source": "Test Organization"
            }
        ]

        # Save sample guidelines
        for guideline in self.sample_guidelines:
            filepath = os.path.join(self.guidelines_dir, f"{guideline['id']}.json")
            with open(filepath, 'w') as f:
                json.dump(guideline, f, indent=2)

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.models.evidence_retrieval.evidence_retriever.SentenceTransformer')
    @patch('src.models.evidence_retrieval.evidence_retriever.faiss')
    def test_initialization(self, mock_faiss, mock_sentence_transformer):
        """Test that the retriever initializes correctly."""
        # Mock the SentenceTransformer
        mock_model_instance = Mock()
        mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model_instance

        # Mock FAISS
        mock_index_instance = Mock()
        mock_index_instance.ntotal = 2
        mock_faiss.IndexFlatIP.return_value = mock_index_instance

        # Create retriever
        retriever = ClinicalGuidelinesEvidenceRetriever(
            guidelines_dir=self.guidelines_dir,
            index_path=self.index_path,
            metadata_path=self.metadata_path
        )

        # Assertions
        assert retriever is not None
        assert retriever.guidelines_dir == self.guidelines_dir
        assert retriever.index_path == self.index_path
        assert retriever.metadata_path == self.metadata_path

        # Check that model was initialized
        mock_sentence_transformer.assert_called_once()

    @patch('src.models.evidence_retrieval.evidence_retriever.SentenceTransformer')
    @patch('src.models.evidence_retrieval.evidence_retriever.faiss')
    def test_retrieve_evidence(self, mock_faiss, mock_sentence_transformer):
        """Test evidence retrieval functionality."""
        # Mock the SentenceTransformer
        mock_model_instance = Mock()
        # Return embeddings for 2 guidelines + 1 query
        mock_model_instance.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),  # guideline embeddings
            np.array([[0.2, 0.3, 0.4]])                    # query embedding
        ]
        mock_sentence_transformer.return_value = mock_model_instance

        # Mock FAISS
        mock_index_instance = Mock()
        mock_index_instance.ntotal = 2
        # Return scores and indices for top 2 results
        mock_index_instance.search.return_value = (
            np.array([[0.8, 0.6]]),  # scores
            np.array([[0, 1]])       # indices
        )
        mock_faiss.IndexFlatIP.return_value = mock_index_instance

        # Create retriever
        retriever = ClinicalGuidelinesEvidenceRetriever(
            guidelines_dir=self.guidelines_dir,
            index_path=self.index_path,
            metadata_path=self.metadata_path
        )

        # Test retrieval
        results = retriever.retrieve_evidence("test query", top_k=2)

        # Assertions
        assert len(results) == 2
        assert results[0]['rank'] == 1
        assert results[0]['similarity_score'] == 0.8
        assert results[1]['rank'] == 2
        assert results[1]['similarity_score'] == 0.6

        # Check that search was called with correct parameters
        mock_index_instance.search.assert_called_once()

    def test_load_sample_guidelines(self):
        """Test loading sample guidelines when none exist."""
        # Create retriever with empty guidelines directory
        empty_dir = os.path.join(self.temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        with patch('src.models.evidence_retrieval.evidence_retriever.SentenceTransformer'), \
             patch('src.models.evidence_retrieval.evidence_retriever.faiss'):

            retriever = ClinicalGuidelinesEvidenceRetriever(
                guidelines_dir=empty_dir,
                index_path=self.index_path,
                metadata_path=self.metadata_path
            )

            # Check that sample guidelines were loaded
            assert len(retriever.guidelines_metadata) > 0
            # Should have loaded our sample guidelines
            guideline_ids = [g['id'] for g in retriever.guidelines_metadata]
            assert 'HTN_001' in guideline_ids  # From sample guidelines

    def test_retrieve_evidence_for_patient(self):
        """Test patient-specific evidence retrieval."""
        with patch('src.models.evidence_retrieval.evidence_retriever.SentenceTransformer') as mock_st, \
             patch('src.models.evidence_retrieval.evidence_retriever.faiss') as mock_faiss:

            # Mock SentenceTransformer
            mock_model_instance = Mock()
            mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_st.return_value = mock_model_instance

            # Mock FAISS
            mock_index_instance = Mock()
            mock_index_instance.ntotal = 2
            mock_index_instance.search.return_value = (
                np.array([[0.8, 0.6]]),
                np.array([[0, 1]])
            )
            mock_faiss.IndexFlatIP.return_value = mock_index_instance

            # Create retriever
            retriever = ClinicalGuidelinesEvidenceRetriever(
                guidelines_dir=self.guidelines_dir,
                index_path=self.index_path,
                metadata_path=self.metadata_path
            )

            # Test patient data
            patient_data = {
                "patient_id": "P00001",
                "age": 65,
                "gender": "Male",
                "conditions": ["hypertension"],
                "adherence_risk": 0.7,
                "discontinuation_risk": 0.2
            }

            # Test retrieval
            results = retriever.retrieve_evidence_for_patient(patient_data, top_k=2)

            # Assertions
            assert isinstance(results, list)
            # Should have attempted to retrieve evidence
            # Note: Actual results depend on mocking


if __name__ == "__main__":
    pytest.main([__file__])