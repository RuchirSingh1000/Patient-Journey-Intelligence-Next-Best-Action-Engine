"""
FastAPI application for PharmaPulse — Patient Journey Intelligence & Next-Best-Action Engine.
Provides REST API endpoints for model inference, recommendations, and evidence retrieval.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modules
try:
    from src.models.next_best_action import NextBestActionEngine
    from src.models.evidence_retrieval.evidence_retriever import ClinicalGuidelinesEvidenceRetriever
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="PharmaPulse API",
    description="Patient Journey Intelligence & Next-Best-Action Engine API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
next_best_action_engine = None
evidence_retriever = None
patient_data = None

# Pydantic models for request/response
class PatientInfo(BaseModel):
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    current_treatment: Optional[int] = None

class RecommendationRequest(BaseModel):
    patient_id: str

class RecommendationResponse(BaseModel):
    patient_id: str
    timestamp: str
    patient_info: Dict[str, Any]
    risk_scores: Dict[str, float]
    recommended_action: str
    action_details: str
    supporting_evidence: Dict[str, Any]

class PopulationRecommendationResponse(BaseModel):
    recommendations: List[RecommendationResponse]
    total_patients: int

class EvidenceQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3

class EvidenceResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    models_loaded: bool
    version: str = "1.0.0"


@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup."""
    global next_best_action_engine, evidence_retriever, patient_data

    print("Initializing PharmaPulse API...")

    if MODELS_AVAILABLE:
        try:
            # Initialize next-best-action engine
            next_best_action_engine = NextBestActionEngine()
            print("Next-best-action engine initialized.")

            # Initialize evidence retriever
            evidence_retriever = ClinicalGuidelinesEvidenceRetriever()
            print("Evidence retriever initialized.")

            # Load patient data for reference
            if next_best_action_engine is not None:
                patient_data = next_best_action_engine.patient_data
                print(f"Loaded data for {len(patient_data)} patients.")

        except Exception as e:
            print(f"Error initializing models: {e}")
            next_best_action_engine = None
            evidence_retriever = None
            patient_data = None
    else:
        print("Models not available. Running in demonstration mode.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to PharmaPulse API - Patient Journey Intelligence & Next-Best-Action Engine",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if MODELS_AVAILABLE else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=MODELS_AVAILABLE
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """Get next-best-action recommendation for a specific patient."""
    if not MODELS_AVAILABLE or next_best_action_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Models not available. Please check server logs."
        )

    try:
        recommendation = next_best_action_engine.recommend_next_best_action(request.patient_id)
        return RecommendationResponse(**recommendation)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/recommendations/population", response_model=PopulationRecommendationResponse)
async def get_population_recommendations(limit: Optional[int] = 100):
    """Get recommendations for a sample of the patient population."""
    if not MODELS_AVAILABLE or next_best_action_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Models not available. Please check server logs."
        )

    try:
        # Get a sample of patients
        sample_size = min(limit, len(next_best_action_engine.patient_data))
        sample_patient_ids = next_best_action_engine.patient_data['patient_id'].head(sample_size).tolist()

        recommendations = []
        for patient_id in sample_patient_ids:
            try:
                rec = next_best_action_engine.recommend_next_best_action(patient_id)
                recommendations.append(RecommendationResponse(**rec))
            except Exception as e:
                print(f"Error generating recommendation for patient {patient_id}: {e}")
                continue

        return PopulationRecommendationResponse(
            recommendations=recommendations,
            total_patients=len(next_best_action_engine.patient_data)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/evidence/retrieve", response_model=EvidenceResponse)
async def retrieve_evidence(query: EvidenceQuery):
    """Retrieve clinical guidelines based on a query."""
    if not MODELS_AVAILABLE or evidence_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Evidence retriever not available. Please check server logs."
        )

    try:
        results = evidence_retriever.retrieve_evidence(query.query, top_k=query.top_k)
        return EvidenceResponse(
            query=query.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/evidence/patient", response_model=EvidenceResponse)
async def retrieve_evidence_for_patient(patient_info: PatientInfo):
    """Retrieve clinical guidelines based on patient information."""
    if not MODELS_AVAILABLE or evidence_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Evidence retriever not available. Please check server logs."
        )

    try:
        # Convert patient info to dict
        patient_dict = patient_info.dict()
        # Remove None values
        patient_dict = {k: v for k, v in patient_dict.items() if v is not None}

        results = evidence_retriever.retrieve_evidence_for_patient(patient_dict, top_k=3)
        return EvidenceResponse(
            query=f"Patient {patient_info.patient_id}",
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/patients/{patient_id}")
async def get_patient_info(patient_id: str):
    """Get basic information for a specific patient."""
    if not MODELS_AVAILABLE or patient_data is None:
        raise HTTPException(
            status_code=503,
            detail="Patient data not available. Please check server logs."
        )

    try:
        patient_row = patient_data[patient_data['patient_id'] == patient_id]
        if len(patient_row) == 0:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

        # Return basic patient info
        patient_info = {
            'patient_id': patient_row['patient_id'].iloc[0],
            'age': int(patient_row['age'].iloc[0]) if 'age' in patient_row.columns else None,
            'gender': patient_row['gender'].iloc[0] if 'gender' in patient_row.columns else None,
            'ethnicity': patient_row['ethnicity'].iloc[0] if 'ethnicity' in patient_row.columns else None,
            'current_treatment': int(patient_row['treatment'].iloc[0]) if 'treatment' in patient_row.columns else None
        }
        return patient_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get basic statistics about the patient population."""
    if not MODELS_AVAILABLE or patient_data is None:
        raise HTTPException(
            status_code=503,
            detail="Patient data not available. Please check server logs."
        )

    try:
        stats = {
            'total_patients': len(patient_data),
            'age_stats': {
                'mean': float(patient_data['age'].mean()) if 'age' in patient_data.columns else None,
                'std': float(patient_data['age'].std()) if 'age' in patient_data.columns else None,
                'min': float(patient_data['age'].min()) if 'age' in patient_data.columns else None,
                'max': float(patient_data['age'].max()) if 'age' in patient_data.columns else None
            },
            'gender_distribution': patient_data['gender'].value_counts().to_dict() if 'gender' in patient_data.columns else {},
            'treatment_distribution': patient_data['treatment'].value_counts().to_dict() if 'treatment' in patient_data.columns else {},
            'cluster_distribution': patient_data['cluster'].value_counts().to_dict() if 'cluster' in patient_data.columns else {}
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )