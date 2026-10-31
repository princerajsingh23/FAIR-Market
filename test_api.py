"""
End-to-End Backend API & User Journey Verification Suite
Tests all 11 endpoints and Journeys A, B, C, D, E against the live FastAPI application.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def run_all_tests():
    print("=== Running FAIR MARKET Backend & ML Verification Suite ===")

    # 1. Health endpoint
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health = r.json()
    assert health["status"] == "healthy"
    assert health["database_records"] == 600
    print(f"[PASS] GET /api/health -> {health['database_records']} properties loaded, R2={health['model_metrics']['r2_score']}")

    # 2. List & Get Properties
    r = client.get("/api/properties?location=Sector%2035,%20Chandigarh&limit=10")
    assert r.status_code == 200
    props = r.json()["properties"]
    assert len(props) > 0
    prop_id = props[0]["id"]
    print(f"[PASS] GET /api/properties -> returned {len(props)} filtered properties")

    r = client.get(f"/api/properties/{prop_id}")
    assert r.status_code == 200
    assert r.json()["id"] == prop_id
    print(f"[PASS] GET /api/properties/{prop_id} -> retrieved property details")

    # 3. Journey A: Predict + Explain + Save
    pred_payload = {
        "location": "Sector 35, Chandigarh",
        "property_type": "Apartment",
        "area_sqft": 1950,
        "bedrooms": 3,
        "bathrooms": 3,
        "property_age": 4,
        "condition": "Excellent",
        "asking_price_inr": 31500000,
    }
    r = client.post("/api/predict", json=pred_payload)
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred = r.json()
    assert pred["estimated_value_inr"] > 10000000
    assert pred["indicative_lower_inr"] < pred["estimated_value_inr"] < pred["indicative_upper_inr"]
    assert len(pred["contributions"]) == 7
    print(
        f"[PASS] POST /api/predict -> Est: INR {int(pred['estimated_value_inr']):,} "
        f"(Range: {int(pred['indicative_lower_inr']):,} - {int(pred['indicative_upper_inr']):,}) | Status: {pred['asking_comparison_label']}"
    )

    # Verify sensitivity: changing inputs changes prediction deterministically
    pred_payload_2 = {**pred_payload, "location": "VIP Road, Zirakpur", "condition": "Average"}
    r2 = client.post("/api/predict", json=pred_payload_2)
    assert r2.status_code == 200
    assert r2.json()["estimated_value_inr"] < pred["estimated_value_inr"]
    print(f"[PASS] Prediction Sensitivity Verified -> Zirakpur Average condition Est: INR {int(r2.json()['estimated_value_inr']):,}")

    # Explain endpoint
    r = client.post("/api/explain", json=pred_payload)
    assert r.status_code == 200
    exp = r.json()
    assert len(exp["positive_factors"]) > 0
    assert "permutation_importances_pct" in exp
    print(f"[PASS] POST /api/explain -> {len(exp['positive_factors'])} positive, {len(exp['negative_factors'])} negative factors")

    # 4. Journey B: Amenities & Location Intelligence
    r = client.get("/api/amenities?location=Sector%2035,%20Chandigarh&categories=Schools,Hospitals,Parks")
    assert r.status_code == 200
    amn = r.json()
    assert amn["location_score"] > 0
    assert len(amn["amenities"]) > 0
    print(f"[PASS] GET /api/amenities -> Location Score: {amn['location_score']}/100, {len(amn['amenities'])} POIs matched")

    # 5. Journey C: Comparables filtering & sorting
    r = client.get("/api/comparables?location=Sector%2035,%20Chandigarh&property_type=Apartment&sort_by=lowest_psf&limit=8")
    assert r.status_code == 200
    comps = r.json()["comparables"]
    assert len(comps) > 0
    assert comps[0]["price_per_sqft"] <= comps[-1]["price_per_sqft"]
    print(f"[PASS] GET /api/comparables -> {len(comps)} sorted comparable apartments returned")

    # 6. Journey D: Future Value Simulator
    sim_payload = {
        "location": "Aerocity, Mohali",
        "property_type": "Villa",
        "area_sqft": 3200,
        "bedrooms": 4,
        "bathrooms": 4,
        "property_age": 5,
        "condition": "Average",
        "renovation_level": "Structural & Interior Upgrade",
        "target_condition": "Excellent",
        "transport_improvement_pct": 12.0,
        "commercial_development_pct": 10.0,
        "civic_green_upgrade_pct": 8.0,
        "preset_name": "Major Renovation + Airport Transit Upgrade",
    }
    r = client.post("/api/simulate", json=sim_payload)
    assert r.status_code == 200
    sim = r.json()
    assert sim["scenario_estimate_inr"] > sim["current_estimate_inr"]
    print(
        f"[PASS] POST /api/simulate -> Current: INR {int(sim['current_estimate_inr']):,} -> "
        f"Scenario: INR {int(sim['scenario_estimate_inr']):,} ({sim['percentage_change']:+.2f}%)"
    )

    # 7. Saved Properties CRUD
    save_payload = {
        "title": "3 BHK Test Residence — Sector 35, Chandigarh",
        "location": "Sector 35, Chandigarh",
        "property_type": "Apartment",
        "area_sqft": 1950,
        "bedrooms": 3,
        "bathrooms": 3,
        "property_age": 4,
        "condition": "Excellent",
        "asking_price_inr": 31500000,
        "estimated_value_inr": pred["estimated_value_inr"],
        "lower_bound_inr": pred["indicative_lower_inr"],
        "upper_bound_inr": pred["indicative_upper_inr"],
        "price_per_sqft": pred["estimated_price_per_sqft"],
        "notes": "Saved during automated E2E test",
    }
    r = client.post("/api/saved-properties", json=save_payload)
    assert r.status_code == 201
    saved_id = r.json()["id"]

    r = client.get("/api/saved-properties")
    assert r.status_code == 200
    assert any(item["id"] == saved_id for item in r.json()["saved_properties"])

    r = client.delete(f"/api/saved-properties/{saved_id}")
    assert r.status_code == 200
    print(f"[PASS] POST/GET/DELETE /api/saved-properties -> persistent CRUD verified ({saved_id})")

    # 8. Journey E: PDF Report Generation
    r = client.post("/api/reports", json=pred_payload)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    print(f"[PASS] POST /api/reports -> Generated valid PDF report ({len(r.content):,} bytes)")

    print("=== ALL 11 ENDPOINTS & JOURNEYS A-E PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_all_tests()
