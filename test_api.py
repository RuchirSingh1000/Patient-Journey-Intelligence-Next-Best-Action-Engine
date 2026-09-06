"""
Test script for PharmaPulse API.
Tests basic API functionality including health check, recommendations, and evidence retrieval.
"""
import requests
import json
import time
import sys
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check() -> Dict[str, Any]:
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health check passed: {data}")
        return data
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return {}

def test_root_endpoint() -> Dict[str, Any]:
    """Test the root endpoint."""
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Root endpoint passed: {data['message']}")
        return data
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return {}

def test_get_patients() -> list:
    """Test getting patient list/statistics."""
    print("Testing patients endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Stats endpoint passed: {data['total_patients']} patients")
        return data
    except Exception as e:
        print(f"✗ Stats endpoint failed: {e}")
        return {}

def test_get_specific_patient() -> Dict[str, Any]:
    """Test getting specific patient info."""
    print("Testing specific patient endpoint...")
    try:
        # First get stats to see what patients are available
        stats_response = requests.get(f"{BASE_URL}/stats")
        stats_response.raise_for_status()
        stats_data = stats_response.json()

        # For now we'll test with a known patient ID from our synthetic data
        patient_id = "P00000"  # This should exist in our generated data

        response = requests.get(f"{BASE_URL}/patients/{patient_id}")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Specific patient endpoint passed: {data}")
        return data
    except Exception as e:
        print(f"✗ Specific patient endpoint failed: {e}")
        return {}

def test_recommendation() -> Dict[str, Any]:
    """Test getting a recommendation for a patient."""
    print("Testing recommendation endpoint...")
    try:
        # Use a known patient ID
        patient_id = "P00000"
        recommendation_request = {"patient_id": patient_id}

        response = requests.post(f"{BASE_URL}/recommend", json=recommendation_request)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Recommendation endpoint passed:")
        print(f"  Patient: {data['patient_id']}")
        print(f"  Action: {data['recommended_action']}")
        print(f"  Adherence Risk: {data['risk_scores']['adherence_risk']:.3f}")
        print(f"  Discontinuation Risk: {data['risk_scores']['discontinuation_risk']:.3f}")
        return data
    except Exception as e:
        print(f"✗ Recommendation endpoint failed: {e}")
        return {}

def test_evidence_retrieval() -> Dict[str, Any]:
    """Test evidence retrieval endpoint."""
    print("Testing evidence retrieval endpoint...")
    try:
        query_request = {
            "query": "patient with hypertension and high medication adherence risk",
            "top_k": 2
        }

        response = requests.post(f"{BASE_URL}/evidence/retrieve", json=query_request)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Evidence retrieval endpoint passed:")
        print(f"  Query: '{data['query']}'")
        print(f"  Results: {data['total_results']}")
        for i, result in enumerate(data['results'][:2], 1):
            print(f"    {i}. {result['title']} (Score: {result['similarity_score']:.3f})")
        return data
    except Exception as e:
        print(f"✗ Evidence retrieval endpoint failed: {e}")
        return {}

def test_evidence_for_patient() -> Dict[str, Any]:
    """Test evidence retrieval for a specific patient."""
    print("Testing evidence for patient endpoint...")
    try:
        patient_info = {
            "patient_id": "P00000",
            "age": 45,
            "gender": "Male",
            "conditions": ["hypertension"]
        }

        response = requests.post(f"{BASE_URL}/evidence/patient", json=patient_info)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Evidence for patient endpoint passed:")
        print(f"  Patient: {data['query']}")
        print(f"  Results: {data['total_results']}")
        for i, result in enumerate(data['results'][:2], 1):
            print(f"    {i}. {result['title']} (Score: {result['similarity_score']:.3f})")
        return data
    except Exception as e:
        print(f"✗ Evidence for patient endpoint failed: {e}")
        return {}

def run_all_tests():
    """Run all API tests."""
    print("=" * 60)
    print("PharmaPulse API Test Suite")
    print("=" * 60)

    # Wait a moment for API to start if needed
    print("Waiting for API to be ready...")
    time.sleep(2)

    tests = [
        test_health_check,
        test_root_endpoint,
        test_get_patients,
        test_get_specific_patient,
        test_recommendation,
        test_evidence_retrieval,
        test_evidence_for_patient
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            result = test()
            if result:  # Non-empty result indicates success
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()

    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)