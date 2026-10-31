"""
FAIR MARKET — FastAPI Application & REST Endpoints
Tagline: Know the value. See the future.
"""

import io
import uuid
import json
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.database.db import init_db, get_conn
from app.schemas.property_schemas import (
    PropertyPredictRequest,
    PredictionResponse,
    SimulationRequest,
    SavedPropertyCreate,
)
from app.services.prediction_service import valuation_service, haversine_km

app = FastAPI(
    title="FAIR MARKET — AI Property Intelligence Platform API",
    description="Explainable AI valuation, hyper-local geospatial intelligence, comparable analysis, and scenario simulation API.",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# Ensure DB is initialized even when imported directly in tests
init_db()


@app.get("/api/health")
def health_check():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM properties")
    prop_cnt = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) as cnt FROM saved_properties")
    saved_cnt = cur.fetchone()["cnt"]
    conn.close()
    return {
        "status": "healthy",
        "service": "FAIR MARKET Property Intelligence API",
        "version": "2.4.0",
        "database_records": prop_cnt,
        "saved_properties_count": saved_cnt,
        "model_metrics": valuation_service.artifact["metrics"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/properties")
def list_properties(
    location: Optional[str] = None,
    property_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    conn = get_conn()
    cur = conn.cursor()
    query = "SELECT * FROM properties WHERE 1=1"
    params = []
    if location and location != "All Locations":
        query += " AND location = ?"
        params.append(location)
    if property_type and property_type != "All":
        query += " AND property_type = ?"
        params.append(property_type)
    if search:
        query += " AND (LOWER(title) LIKE ? OR LOWER(location) LIKE ? OR LOWER(property_type) LIKE ?)"
        q = f"%{search.lower()}%"
        params.extend([q, q, q])
    query += " ORDER BY analysed_date DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"count": len(rows), "properties": rows, "data_notice": "SYNTHETIC DEMO DATA"}


@app.get("/api/properties/{property_id}")
def get_property(property_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM properties WHERE id = ?", (property_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Property '{property_id}' not found")
    return dict(row)


@app.post("/api/predict", response_model=PredictionResponse)
def predict_property_value(payload: PropertyPredictRequest):
    result = valuation_service.compute_prediction_and_explanation(payload)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (
            id, location, property_type, area_sqft, bedrooms, bathrooms,
            property_age, condition, asking_price_inr, estimated_value_inr,
            lower_bound_inr, upper_bound_inr, estimated_psf, valuation_status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["prediction_id"],
            payload.location,
            payload.property_type,
            payload.area_sqft,
            payload.bedrooms,
            payload.bathrooms,
            payload.property_age,
            payload.condition,
            payload.asking_price_inr,
            result["estimated_value_inr"],
            result["indicative_lower_inr"],
            result["indicative_upper_inr"],
            result["estimated_price_per_sqft"],
            result["asking_comparison_label"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()
    return result


@app.post("/api/explain")
def explain_property_value(payload: PropertyPredictRequest):
    result = valuation_service.compute_prediction_and_explanation(payload)
    pos = [c for c in result["contributions"] if c["impact_inr"] > 0]
    neg = [c for c in result["contributions"] if c["impact_inr"] < 0]
    return {
        "prediction_id": result["prediction_id"],
        "estimated_value_inr": result["estimated_value_inr"],
        "positive_factors": pos,
        "negative_factors": neg,
        "all_contributions": result["contributions"],
        "permutation_importances_pct": result["model_info"]["permutation_importances_pct"],
        "plain_english_summary": result["plain_english_summary"],
        "methodology_note": result["model_info"]["explainability_method"],
    }


@app.get("/api/comparables")
def get_comparable_properties(
    location: str = Query("Sector 35, Chandigarh"),
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_area: Optional[int] = None,
    max_area: Optional[int] = None,
    bedrooms: Optional[int] = None,
    max_distance_km: float = Query(15.0, ge=0.5, le=50.0),
    sort_by: str = Query("nearest"),
    limit: int = Query(24, ge=1, le=100),
):
    loc_meta = valuation_service.localities.get(location, list(valuation_service.localities.values())[0])
    ref_lat, ref_lng = loc_meta["lat"], loc_meta["lng"]

    conn = get_conn()
    cur = conn.cursor()
    query = "SELECT * FROM properties WHERE 1=1"
    params = []
    if property_type and property_type != "All":
        query += " AND property_type = ?"
        params.append(property_type)
    if min_price is not None:
        query += " AND asking_price_inr >= ?"
        params.append(min_price)
    if max_price is not None:
        query += " AND asking_price_inr <= ?"
        params.append(max_price)
    if min_area is not None:
        query += " AND area_sqft >= ?"
        params.append(min_area)
    if max_area is not None:
        query += " AND area_sqft <= ?"
        params.append(max_area)
    if bedrooms is not None and bedrooms >= 0:
        query += " AND bedrooms = ?"
        params.append(bedrooms)

    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    enriched = []
    for r in rows:
        dist = round(haversine_km(ref_lat, ref_lng, r["latitude"], r["longitude"]), 2)
        if dist <= max_distance_km:
            r["distance_km"] = dist
            enriched.append(r)

    if sort_by == "lowest_price":
        enriched.sort(key=lambda x: x["asking_price_inr"])
    elif sort_by == "highest_price":
        enriched.sort(key=lambda x: x["asking_price_inr"], reverse=True)
    elif sort_by == "lowest_psf":
        enriched.sort(key=lambda x: x["price_per_sqft"])
    elif sort_by == "highest_psf":
        enriched.sort(key=lambda x: x["price_per_sqft"], reverse=True)
    else:
        enriched.sort(key=lambda x: x["distance_km"])

    return {
        "reference_location": location,
        "reference_coordinates": {"lat": ref_lat, "lng": ref_lng},
        "count": len(enriched[:limit]),
        "comparables": enriched[:limit],
    }


@app.get("/api/amenities")
def get_nearby_amenities(
    location: str = Query("Sector 35, Chandigarh"),
    categories: Optional[str] = Query(None, description="Comma-separated categories: Schools,Hospitals,Parks,Markets,Transport"),
    radius_km: float = Query(12.0, ge=0.5, le=50.0),
):
    loc_meta = valuation_service.localities.get(location, list(valuation_service.localities.values())[0])
    ref_lat, ref_lng = loc_meta["lat"], loc_meta["lng"]

    active_cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else ["Schools", "Hospitals", "Parks", "Markets", "Transport"]

    items = []
    counts = {c: 0 for c in ["Schools", "Hospitals", "Parks", "Markets", "Transport"]}
    min_dists = {c: 99.0 for c in ["Schools", "Hospitals", "Parks", "Markets", "Transport"]}

    for amn in valuation_service.amenities:
        d = round(haversine_km(ref_lat, ref_lng, amn["lat"], amn["lng"]), 2)
        cat = amn["category"]
        if d < min_dists[cat]:
            min_dists[cat] = d
        if d <= radius_km and cat in active_cats:
            counts[cat] = counts.get(cat, 0) + 1
            items.append({**amn, "distance_km": d})

    items.sort(key=lambda x: x["distance_km"])

    # Transparent location score calculation
    cat_scores = {}
    for c, md in min_dists.items():
        cat_scores[c] = round(min(100.0, max(25.0, 100.0 * (2.71828 ** (-0.11 * max(0.0, md - 0.3))))), 1)
    overall_score = round(sum(cat_scores.values()) / len(cat_scores), 1)

    return {
        "selected_location": location,
        "coordinates": {"lat": ref_lat, "lng": ref_lng},
        "location_score": overall_score,
        "category_scores": cat_scores,
        "formula_explanation": "Location Score = Mean across 5 categories of 100 * exp(-0.11 * max(0, nearest_distance_km - 0.3)).",
        "counts_by_category": counts,
        "amenities": items,
    }


@app.post("/api/simulate")
def simulate_future_value(payload: SimulationRequest):
    res = valuation_service.simulate_scenario(payload)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO simulation_scenarios (
            id, property_location, preset_name, base_estimate_inr,
            scenario_estimate_inr, pct_change, assumptions_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"SIM-{uuid.uuid4().hex[:8].upper()}",
            payload.location,
            payload.preset_name,
            res["current_estimate_inr"],
            res["scenario_estimate_inr"],
            res["percentage_change"],
            json.dumps(res["documented_assumptions"]),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()
    return res


@app.get("/api/saved-properties")
def list_saved_properties(search: Optional[str] = None, property_type: Optional[str] = None):
    conn = get_conn()
    cur = conn.cursor()
    query = "SELECT * FROM saved_properties WHERE 1=1"
    params = []
    if property_type and property_type != "All":
        query += " AND property_type = ?"
        params.append(property_type)
    if search:
        query += " AND (LOWER(title) LIKE ? OR LOWER(location) LIKE ?)"
        q = f"%{search.lower()}%"
        params.extend([q, q])
    query += " ORDER BY saved_at DESC"
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"count": len(rows), "saved_properties": rows}


@app.post("/api/saved-properties", status_code=201)
def create_saved_property(payload: SavedPropertyCreate):
    new_id = f"SAVED-{uuid.uuid4().hex[:6].upper()}"
    saved_at = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO saved_properties (
            id, title, location, property_type, area_sqft, bedrooms, bathrooms,
            property_age, condition, asking_price_inr, estimated_value_inr,
            lower_bound_inr, upper_bound_inr, price_per_sqft, notes, saved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_id,
            payload.title,
            payload.location,
            payload.property_type,
            payload.area_sqft,
            payload.bedrooms,
            payload.bathrooms,
            payload.property_age,
            payload.condition,
            payload.asking_price_inr,
            payload.estimated_value_inr,
            payload.lower_bound_inr,
            payload.upper_bound_inr,
            payload.price_per_sqft,
            payload.notes or "",
            saved_at,
        ),
    )
    conn.commit()
    conn.close()
    return {"id": new_id, "saved_at": saved_at, **payload.dict()}


@app.delete("/api/saved-properties/{property_id}")
def delete_saved_property(property_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM saved_properties WHERE id = ?", (property_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Saved property '{property_id}' not found")
    cur.execute("DELETE FROM saved_properties WHERE id = ?", (property_id,))
    conn.commit()
    conn.close()
    return {"deleted": True, "id": property_id}


@app.post("/api/reports")
def generate_valuation_report(payload: PropertyPredictRequest):
    analysis = valuation_service.compute_prediction_and_explanation(payload)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Header banner
    c.setFillColorRGB(0.043, 0.071, 0.125)
    c.rect(0, height - 95, width, 95, fill=1, stroke=0)
    c.setFillColorRGB(0.208, 0.839, 0.898)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 42, "FAIR MARKET — AI PROPERTY INTELLIGENCE REPORT")
    c.setFillColorRGB(0.96, 0.97, 0.99)
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 60, "Tagline: Know the value. See the future. | Indicative ML Valuation & XAI Dossier")
    c.drawString(40, height - 76, f"Report ID: {analysis['prediction_id']}   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")

    y = height - 125
    c.setFillColorRGB(0.09, 0.12, 0.20)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "1. Subject Property Specification & Model Estimate")
    y -= 20
    c.setFont("Helvetica", 10)
    specs = [
        f"Locality: {payload.location}   |   Property Type: {payload.property_type}",
        f"Built-up Area: {payload.area_sqft:,} sq.ft.   |   Configuration: {payload.bedrooms} Bed / {payload.bathrooms} Bath",
        f"Property Age: {payload.property_age} Years   |   Condition: {payload.condition}",
        f"Estimated Market Value: INR {int(analysis['estimated_value_inr']):,}  (INR {int(analysis['estimated_price_per_sqft']):,}/sq.ft.)",
        f"Indicative Range (10th-90th Test Residual): INR {int(analysis['indicative_lower_inr']):,} to INR {int(analysis['indicative_upper_inr']):,}",
        f"Asking Price Comparison: {analysis['asking_comparison_label']}",
    ]
    for line in specs:
        c.drawString(48, y, line)
        y -= 16

    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "2. Explainable AI (SHAP / Additive Feature Contributions)")
    y -= 20
    c.setFont("Helvetica", 9.5)
    for contrib in analysis["contributions"]:
        sign = "+" if contrib["impact_inr"] >= 0 else "-"
        c.drawString(
            48,
            y,
            f"• {contrib['display_name']} [{contrib['input_value']}]: {sign}INR {abs(int(contrib['impact_inr'])):,} ({contrib['impact_pct']:+.1f}%)",
        )
        y -= 14

    y -= 12
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "3. Data Provenance & Methodology Limitations")
    y -= 18
    c.setFont("Helvetica", 9)
    c.drawString(48, y, "Dataset: SYNTHETIC DEMO DATA (Chandigarh-Mohali-Kharar-Zirakpur 600-record calibrated cohort).")
    y -= 14
    c.drawString(48, y, f"Model: {analysis['model_info']['algorithm']} | Out-of-Sample Test R2 = {analysis['model_info']['r2_score']}.")
    y -= 14
    c.drawString(48, y, "Disclaimer: This report is an indicative algorithmic estimate for educational/decision-support use only,")
    y -= 14
    c.drawString(48, y, "and does not constitute a certified valuation or bank appraisal.")

    c.showPage()
    c.save()
    pdf_bytes = buf.getvalue()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="FAIR_MARKET_Report_{analysis["prediction_id"]}.pdf"'},
    )
