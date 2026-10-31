"""
FAIR MARKET — Database Persistence Layer
Supports PostgreSQL via SQLAlchemy (DATABASE_URL) in production and SQLite persistent database fallback
for zero-config local/sandbox execution.
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional

DB_PATH = os.getenv(
    "SQLITE_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fair_market.db"),
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS properties (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    location TEXT NOT NULL,
    city TEXT NOT NULL,
    property_type TEXT NOT NULL,
    area_sqft INTEGER NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms INTEGER NOT NULL,
    property_age INTEGER NOT NULL,
    condition TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    amenity_score REAL NOT NULL,
    infra_score REAL NOT NULL,
    connectivity_index REAL NOT NULL,
    commercial_index REAL NOT NULL,
    price_inr REAL NOT NULL,
    asking_price_inr REAL NOT NULL,
    price_per_sqft REAL NOT NULL,
    data_source TEXT NOT NULL,
    analysed_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    property_type TEXT NOT NULL,
    area_sqft INTEGER NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms INTEGER NOT NULL,
    property_age INTEGER NOT NULL,
    condition TEXT NOT NULL,
    asking_price_inr REAL,
    estimated_value_inr REAL NOT NULL,
    lower_bound_inr REAL NOT NULL,
    upper_bound_inr REAL NOT NULL,
    estimated_psf REAL NOT NULL,
    valuation_status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS saved_properties (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    location TEXT NOT NULL,
    property_type TEXT NOT NULL,
    area_sqft INTEGER NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms INTEGER NOT NULL,
    property_age INTEGER NOT NULL,
    condition TEXT NOT NULL,
    asking_price_inr REAL,
    estimated_value_inr REAL NOT NULL,
    lower_bound_inr REAL NOT NULL,
    upper_bound_inr REAL NOT NULL,
    price_per_sqft REAL NOT NULL,
    notes TEXT,
    saved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulation_scenarios (
    id TEXT PRIMARY KEY,
    property_location TEXT NOT NULL,
    preset_name TEXT NOT NULL,
    base_estimate_inr REAL NOT NULL,
    scenario_estimate_inr REAL NOT NULL,
    pct_change REAL NOT NULL,
    assumptions_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    conn.commit()

    # Seed properties from tricity_properties_demo.json if empty
    cur.execute("SELECT COUNT(*) as cnt FROM properties")
    if cur.fetchone()["cnt"] == 0:
        data_file = os.path.join(os.path.dirname(DB_PATH), "tricity_properties_demo.json")
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            cur.executemany(
                """
                INSERT INTO properties (
                    id, title, location, city, property_type, area_sqft, bedrooms, bathrooms,
                    property_age, condition, latitude, longitude, amenity_score, infra_score,
                    connectivity_index, commercial_index, price_inr, asking_price_inr,
                    price_per_sqft, data_source, analysed_date
                ) VALUES (
                    :id, :title, :location, :city, :property_type, :area_sqft, :bedrooms, :bathrooms,
                    :property_age, :condition, :latitude, :longitude, :amenity_score, :infra_score,
                    :connectivity_index, :commercial_index, :price_inr, :asking_price_inr,
                    :price_per_sqft, :data_source, :analysed_date
                )
                """,
                records,
            )
            conn.commit()

    # Seed 3 initial saved properties if empty
    cur.execute("SELECT COUNT(*) as cnt FROM saved_properties")
    if cur.fetchone()["cnt"] == 0:
        initial_saved = [
            {
                "id": "SAVED-101",
                "title": "3 BHK Luxury Apartment — Sector 35, Chandigarh",
                "location": "Sector 35, Chandigarh",
                "property_type": "Apartment",
                "area_sqft": 1850,
                "bedrooms": 3,
                "bathrooms": 3,
                "property_age": 3,
                "condition": "Excellent",
                "asking_price_inr": 29200000.0,
                "estimated_value_inr": 28450000.0,
                "lower_bound_inr": 26515000.0,
                "upper_bound_inr": 30498000.0,
                "price_per_sqft": 15378.0,
                "notes": "Corner park-facing unit near Sector 35C Promenade Market.",
                "saved_at": "2026-10-28",
            },
            {
                "id": "SAVED-102",
                "title": "4 BHK Contemporary Villa — Aerocity, Mohali",
                "location": "Aerocity, Mohali",
                "property_type": "Villa",
                "area_sqft": 3400,
                "bedrooms": 4,
                "bathrooms": 5,
                "property_age": 1,
                "condition": "Excellent",
                "asking_price_inr": 45500000.0,
                "estimated_value_inr": 46800000.0,
                "lower_bound_inr": 43618000.0,
                "upper_bound_inr": 50170000.0,
                "price_per_sqft": 13765.0,
                "notes": "Direct connectivity to Shaheed Bhagat Singh Intl Airport.",
                "saved_at": "2026-10-29",
            },
            {
                "id": "SAVED-103",
                "title": "3 BHK High-Rise Residence — PR-7 Airport Road, Zirakpur",
                "location": "PR-7 Airport Road, Zirakpur",
                "property_type": "Apartment",
                "area_sqft": 1650,
                "bedrooms": 3,
                "bathrooms": 3,
                "property_age": 2,
                "condition": "Good",
                "asking_price_inr": 10900000.0,
                "estimated_value_inr": 10680000.0,
                "lower_bound_inr": 9954000.0,
                "upper_bound_inr": 11449000.0,
                "price_per_sqft": 6473.0,
                "notes": "Gated society along PR-7 express corridor.",
                "saved_at": "2026-10-30",
            },
        ]
        cur.executemany(
            """
            INSERT INTO saved_properties (
                id, title, location, property_type, area_sqft, bedrooms, bathrooms,
                property_age, condition, asking_price_inr, estimated_value_inr,
                lower_bound_inr, upper_bound_inr, price_per_sqft, notes, saved_at
            ) VALUES (
                :id, :title, :location, :property_type, :area_sqft, :bedrooms, :bathrooms,
                :property_age, :condition, :asking_price_inr, :estimated_value_inr,
                :lower_bound_inr, :upper_bound_inr, :price_per_sqft, :notes, :saved_at
            )
            """,
            initial_saved,
        )
        conn.commit()
    conn.close()
