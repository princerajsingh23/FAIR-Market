"""
FAIR MARKET — Pydantic Request & Response Schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class PropertyPredictRequest(BaseModel):
    location: str = Field(..., description="Tricity locality name")
    property_type: str = Field(..., description="Apartment, Independent House, Villa, Residential Plot, or Commercial Property")
    area_sqft: int = Field(..., ge=250, le=25000, description="Built-up or plot area in sq. ft.")
    bedrooms: int = Field(0, ge=0, le=12, description="Number of bedrooms (0 for plots/commercial)")
    bathrooms: int = Field(0, ge=0, le=12, description="Number of bathrooms")
    property_age: int = Field(0, ge=0, le=80, description="Property age in years")
    condition: str = Field(..., description="Excellent, Good, Average, or Needs Renovation")
    asking_price_inr: Optional[float] = Field(None, ge=100000, le=1000000000, description="Optional seller asking price in INR")

    @validator("property_type")
    def validate_property_type(cls, v):
        allowed = {"Apartment", "Independent House", "Villa", "Residential Plot", "Commercial Property"}
        if v not in allowed:
            raise ValueError(f"property_type must be one of {sorted(allowed)}")
        return v

    @validator("condition")
    def validate_condition(cls, v):
        allowed = {"Excellent", "Good", "Average", "Needs Renovation"}
        if v not in allowed:
            raise ValueError(f"condition must be one of {sorted(allowed)}")
        return v


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    category: str
    input_value: str
    impact_inr: float
    impact_pct: float
    direction: str
    explanation: str


class PredictionResponse(BaseModel):
    prediction_id: str
    estimated_value_inr: float
    indicative_lower_inr: float
    indicative_upper_inr: float
    estimated_price_per_sqft: float
    asking_price_inr: Optional[float]
    asking_vs_estimated_diff_inr: Optional[float]
    asking_vs_estimated_pct: Optional[float]
    asking_comparison_label: str
    location_coordinates: Dict[str, float]
    model_info: Dict[str, Any]
    contributions: List[FeatureContribution]
    plain_english_summary: List[str]


class SimulationRequest(BaseModel):
    location: str
    property_type: str
    area_sqft: int = Field(..., ge=250, le=25000)
    bedrooms: int = Field(0, ge=0, le=12)
    bathrooms: int = Field(0, ge=0, le=12)
    property_age: int = Field(0, ge=0, le=80)
    condition: str
    renovation_level: str = Field("None", description="None, Basic Cosmetic, Structural & Interior Upgrade, Smart Luxury Overhaul")
    target_condition: str = Field(..., description="Target condition after renovation")
    transport_improvement_pct: float = Field(0.0, ge=0.0, le=25.0)
    commercial_development_pct: float = Field(0.0, ge=0.0, le=25.0)
    civic_green_upgrade_pct: float = Field(0.0, ge=0.0, le=15.0)
    preset_name: str = Field("Custom Scenario")


class SavedPropertyCreate(BaseModel):
    title: str
    location: str
    property_type: str
    area_sqft: int = Field(..., ge=250)
    bedrooms: int = Field(0, ge=0)
    bathrooms: int = Field(0, ge=0)
    property_age: int = Field(0, ge=0)
    condition: str
    asking_price_inr: Optional[float] = None
    estimated_value_inr: float
    lower_bound_inr: float
    upper_bound_inr: float
    price_per_sqft: float
    notes: Optional[str] = ""
