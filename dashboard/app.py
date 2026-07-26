"""
FastAPI Dashboard Backend
========================
Serves the analyst dashboard UI and API endpoints for:
- Alert Queue (paginated, filterable by risk, entity, attack type, cold-start, drift)
- Alert Detail (JSON breakdown, feature attribution, explanation)
- Entity History (profile card, timeline events, past alerts)
- Evaluation Metrics (PR AUC, precision@alert-budget, figures)
"""

from _pytest import monkeypatch
import json
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "templates")
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")

# Ensure required directories exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

app = FastAPI(
    title="Honeywell Sentinel AI — Threat Anomaly Intelligence",
    version="1.0.0",
    description="Enterprise Cybersecurity Behavioral Anomaly Detection & SOC Analyst Interface"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Data Caching Helper
_cache = {}

def get_data_file(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    mtime = os.path.getmtime(path)
    if filename not in _cache or _cache[filename]["mtime"] != mtime:
        df = pd.read_csv(path)
        df = df.where(pd.notnull(df), None)
        _cache[filename] = {"mtime": mtime, "df": df}
    return _cache[filename]["df"]


def clean_record(obj):
    """Recursively replace float('nan') and float('inf') with None for clean JSON responses."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_record(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_record(v) for v in obj]
    return obj



@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/stats")
async def get_stats():
    alerts_df = get_data_file("alerts.csv")
    scored_df = get_data_file("scored_events.csv")
    profiles_df = get_data_file("entity_profiles.csv")

    total_events = len(scored_df)
    total_alerts = len(alerts_df)
    total_entities = len(profiles_df)

    attack_distribution = (
        alerts_df["predicted_type"].value_counts().to_dict()
        if not alerts_df.empty and "predicted_type" in alerts_df.columns
        else {}
    )

    critical_count = (
        (alerts_df["risk_score"] >= 0.40).sum()
        if not alerts_df.empty and "risk_score" in alerts_df.columns
        else 0
    )

    cold_start_alerts = (
        alerts_df["is_cold_start"].astype(bool).sum()
        if not alerts_df.empty and "is_cold_start" in alerts_df.columns
        else 0
    )

    return JSONResponse({
        "total_events_analyzed": total_events,
        "total_alerts": total_alerts,
        "total_entities": total_entities,
        "critical_alerts_count": int(critical_count),
        "cold_start_alerts_count": int(cold_start_alerts),
        "attack_type_breakdown": attack_distribution,
    })


@app.get("/api/alerts")
async def get_alerts(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
    attack_type: Optional[str] = None,
    search: Optional[str] = None,
    top_pct: Optional[float] = Query(None, ge=0.1, le=100.0)
):
    df = get_data_file("alerts.csv")
    if df.empty:
        return JSONResponse({"total": 0, "page": page, "limit": limit, "alerts": []})

    # Alert Budget filter (top X% riskiest events)
    if top_pct is not None:
        scored_df = get_data_file("scored_events.csv")
        if not scored_df.empty:
            n_top = max(1, int(len(scored_df) * (top_pct / 100.0)))
            top_risk_threshold = scored_df["risk_score"].nlargest(n_top).min()
            df = df[df["risk_score"] >= top_risk_threshold]

    # Filters
    if min_risk > 0:
        df = df[df["risk_score"] >= min_risk]

    if attack_type and attack_type != "all":
        df = df[df["predicted_type"] == attack_type]

    if search:
        search_lower = search.lower()
        mask = (
            df["entity_id"].astype(str).str.lower().str.contains(search_lower) |
            df["source_ip"].astype(str).str.lower().str.contains(search_lower) |
            df["resource_accessed"].astype(str).str.lower().str.contains(search_lower)
        )
        df = df[mask]

    # Sort descending by risk score
    df = df.sort_values("risk_score", ascending=False)
    total_records = len(df)

    start = (page - 1) * limit
    end = start + limit
    paginated_df = df.iloc[start:end].copy()

    records = paginated_df.to_dict(orient="records")
    for r in records:
        if isinstance(r.get("top_features"), str):
            try:
                r["top_features"] = json.loads(r["top_features"])
            except:
                r["top_features"] = []

    records = clean_record(records)

    return JSONResponse({
        "total": total_records,
        "page": page,
        "limit": limit,
        "alerts": records,
    })


@app.get("/api/alerts/export")
async def export_alerts(
    top_pct: Optional[float] = Query(None, ge=0.1, le=100.0),
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
    attack_type: Optional[str] = None,
    search: Optional[str] = None
):
    """Export filtered alerts queue to downloadable CSV."""
    df = get_data_file("alerts.csv")
    if df.empty:
        return Response(content="no data", media_type="text/csv")

    # Alert Budget filter (top X% riskiest events) matching get_alerts algorithm
    if top_pct is not None:
        scored_df = get_data_file("scored_events.csv")
        if not scored_df.empty:
            n_top = max(1, int(len(scored_df) * (top_pct / 100.0)))
            top_risk_threshold = scored_df["risk_score"].nlargest(n_top).min()
            df = df[df["risk_score"] >= top_risk_threshold]

    if min_risk > 0:
        df = df[df["risk_score"] >= min_risk]

    if attack_type and attack_type != "all":
        df = df[df["predicted_type"] == attack_type]

    if search:
        s_lower = search.lower()
        df = df[
            df["entity_id"].astype(str).str.lower().str.contains(s_lower) |
            df["source_ip"].astype(str).str.lower().str.contains(s_lower) |
            df["resource_accessed"].astype(str).str.lower().str.contains(s_lower)
        ]

    df = df.sort_values(by="risk_score", ascending=False)
    csv_data = df.to_csv(index=False)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=soc_alerts_report.csv"}
    )


@app.get("/api/alerts/{session_id}")
async def get_alert_detail(session_id: str):
    df = get_data_file("alerts.csv")
    if df.empty:
        raise HTTPException(status_code=404, detail="Alerts database empty")

    match = df[df["session_id"] == session_id]
    if match.empty:
        scored_df = get_data_file("scored_events.csv")
        match = scored_df[scored_df["session_id"] == session_id]
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Alert with session_id '{session_id}' not found")

    record = match.iloc[0].to_dict()
    if isinstance(record.get("top_features"), str):
        try:
            record["top_features"] = json.loads(record["top_features"])
        except:
            record["top_features"] = []

    record = clean_record(record)
    return JSONResponse(record)


@app.get("/api/entities")
async def get_entities(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    entity_type: Optional[str] = None,
    search: Optional[str] = None
):
    profiles_df = get_data_file("entity_profiles.csv")
    alerts_df = get_data_file("alerts.csv")

    if profiles_df.empty:
        return JSONResponse({"total": 0, "page": page, "limit": limit, "entities": []})

    alert_counts = (
        alerts_df["entity_id"].value_counts().to_dict()
        if not alerts_df.empty else {}
    )

    records = profiles_df.to_dict(orient="records")
    for r in records:
        eid = r["entity_id"]
        r["alert_count"] = alert_counts.get(eid, 0)
        for field in ["home_geo_set", "home_ip_prefixes", "known_device_fingerprints", "typical_resources"]:
            if isinstance(r.get(field), str):
                try:
                    r[field] = json.loads(r[field])
                except:
                    pass

    if entity_type and entity_type != "all":
        records = [r for r in records if r.get("entity_type") == entity_type]

    if search:
        search_lower = search.lower()
        records = [
            r for r in records
            if search_lower in str(r["entity_id"]).lower() or search_lower in str(r["entity_type"]).lower()
        ]

    total_records = len(records)
    start = (page - 1) * limit
    end = start + limit
    paginated_records = records[start:end]

    return JSONResponse({
        "total": total_records,
        "page": page,
        "limit": limit,
        "entities": paginated_records
    })


@app.get("/api/entities/{entity_id}")
async def get_entity_detail(entity_id: str):
    profiles_df = get_data_file("entity_profiles.csv")
    scored_df = get_data_file("scored_events.csv")
    alerts_df = get_data_file("alerts.csv")

    p_match = profiles_df[profiles_df["entity_id"] == entity_id]
    if p_match.empty:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    profile_dict = p_match.iloc[0].to_dict()
    for field in ["home_geo_set", "home_ip_prefixes", "known_device_fingerprints", "typical_resources"]:
        if isinstance(profile_dict.get(field), str):
            try:
                profile_dict[field] = json.loads(profile_dict[field])
            except:
                pass

    entity_events = scored_df[scored_df["entity_id"] == entity_id].copy()
    if not entity_events.empty:
        entity_events = entity_events.sort_values("timestamp").tail(100)
        events_list = entity_events.to_dict(orient="records")
    else:
        events_list = []

    entity_alerts = (
        alerts_df[alerts_df["entity_id"] == entity_id].to_dict(orient="records")
        if not alerts_df.empty else []
    )

    return JSONResponse({
        "profile": clean_record(profile_dict),
        "recent_events": clean_record(events_list),
        "past_alerts": clean_record(entity_alerts)
    })




@app.post("/api/simulate-attack")
async def simulate_attack(type: str = "brute_force"):
    """Simulate a live cyber attack injection for live demonstration."""
    import uuid
    from datetime import datetime

    alerts_df = get_data_file("alerts.csv")

    sim_id = str(uuid.uuid4())[:12]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    patterns = {
        "brute_force": {
            "source_ip": "198.51.100.44",
            "geo_location": "Moscow, RU",
            "resource_accessed": "/admin/auth-token-issue",
            "risk_score": 0.8950,
            "explanation": "LIVE SIMULATION: 48 rapid failed authentication attempts detected from IP 198.51.100.44 within 30 seconds (brute_force pattern).",
            "top_features": [["failed_auth_count_5min", 48.0], ["baseline_score", 0.785], ["geo_novelty", 1.0]]
        },
        "impossible_travel": {
            "source_ip": "185.220.101.5",
            "geo_location": "Frankfurt, DE",
            "resource_accessed": "/vpn/gateway-connect",
            "risk_score": 0.9420,
            "explanation": "LIVE SIMULATION: Impossible travel alert flagged for user sim_user_999 from Frankfurt, DE just 4 minutes after login from New York, US.",
            "top_features": [["geo_novelty", 1.0], ["hour_zscore", 3.8], ["baseline_score", 0.892]]
        },
        "credential_stuffing": {
            "source_ip": "103.251.170.8",
            "geo_location": "Hanoi, VN",
            "resource_accessed": "/api/v1/auth/login",
            "risk_score": 0.8730,
            "explanation": "LIVE SIMULATION: Multi-account credential stuffing detected across 32 user accounts from single source IP within 2 minutes.",
            "top_features": [["distinct_entities_from_ip_5min", 32.0], ["failed_auth_count_5min", 35.0], ["geo_novelty", 1.0]]
        },
        "lateral_movement": {
            "source_ip": "10.0.4.112",
            "geo_location": "Internal Subnet (Zone B)",
            "resource_accessed": "/sys/kerberos/admin-ticket-grant",
            "risk_score": 0.9150,
            "explanation": "LIVE SIMULATION: Unusually high privilege Kerberos ticket request from internal endpoint 10.0.4.112 targeting domain controller.",
            "top_features": [["resource_novelty", 1.0], ["baseline_score", 0.912], ["hour_zscore", 2.9]]
        }
    }

    selected_pattern = patterns.get(type, patterns["brute_force"])

    sim_alert = {
        "session_id": sim_id,
        "entity_id": "sim_user_999",
        "entity_type": "user",
        "timestamp": now_str,
        "source_ip": selected_pattern["source_ip"],
        "geo_location": selected_pattern["geo_location"],
        "resource_accessed": selected_pattern["resource_accessed"],
        "auth_method": "password",
        "auth_result": "failure",
        "session_duration": 0.0,
        "device_fingerprint": "dev_sim_999",
        "risk_score": selected_pattern["risk_score"],
        "predicted_type": type if type in patterns else "brute_force",
        "predicted_type_confidence": 0.98,
        "explanation_text": selected_pattern["explanation"],
        "top_features": json.dumps(selected_pattern["top_features"]),
        "is_cold_start": False,
        "is_drift_flagged": False
    }

    sim_df = pd.DataFrame([sim_alert])
    updated_alerts = pd.concat([sim_df, alerts_df], ignore_index=True)

    # Update in-memory cache directly
    _cache["alerts.csv"] = {
        "mtime": 9999999999.0,
        "df": updated_alerts
    }

    # Attempt file write (catch any file lock errors on Windows gracefully)
    try:
        alerts_path = os.path.join(DATA_DIR, "alerts.csv")
        updated_alerts.to_csv(alerts_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save alerts.csv to disk ({e}), using in-memory update.")

    return JSONResponse({
        "status": "success",
        "message": f"Live {type.replace('_', ' ').title()} attack injected successfully!",
        "simulated_alert": clean_record(sim_alert)
    })


@app.get("/api/metrics")
async def get_metrics():
    metrics_path = os.path.join(DATA_DIR, "metrics_summary.json")
    if not os.path.exists(metrics_path):
        return JSONResponse({"status": "not_evaluated_yet"})
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return JSONResponse(data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
