"""
Dashboard API Endpoint Test
===========================
Tests all FastAPI endpoints to ensure 200 OK responses and proper JSON payloads.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_endpoints():
    print("=" * 60)
    print("TESTING DASHBOARD FASTAPI ENDPOINTS")
    print("=" * 60)

    # 1. Test HTML Dashboard root
    response = client.get("/")
    assert response.status_code == 200, f"Root / failed: {response.status_code}"
    print("[OK] GET / -> 200 OK (HTML Template)")

    # 1b. Test Static Assets
    res_css = client.get("/static/css/styles.css")
    assert res_css.status_code == 200, "CSS failed to load"
    res_js = client.get("/static/js/app.js")
    assert res_js.status_code == 200, "JS failed to load"
    print("[OK] GET /static/css/styles.css & /static/js/app.js -> 200 OK")

    # 2. Test Stats API
    response = client.get("/api/stats")
    assert response.status_code == 200, f"/api/stats failed: {response.status_code}"
    stats = response.json()
    assert "total_events_analyzed" in stats, "Missing total_events_analyzed in stats"
    print(f"[OK] GET /api/stats -> 200 OK (Analyzed Events: {stats['total_events_analyzed']}, Alerts: {stats['total_alerts']})")

    # 3. Test Alerts API
    response = client.get("/api/alerts?page=1&limit=10")
    assert response.status_code == 200, f"/api/alerts failed: {response.status_code}"
    alerts_data = response.json()
    assert "alerts" in alerts_data, "Missing alerts in response"
    alerts = alerts_data["alerts"]
    print(f"[OK] GET /api/alerts -> 200 OK (Retrieved {len(alerts)} alerts, Total: {alerts_data['total']})")

    # 4. Test Alert Detail API
    if alerts:
        session_id = alerts[0]["session_id"]
        response = client.get(f"/api/alerts/{session_id}")
        assert response.status_code == 200, f"/api/alerts/{session_id} failed: {response.status_code}"
        detail = response.json()
        assert "explanation_text" in detail, "Missing explanation_text"
        print(f"[OK] GET /api/alerts/{session_id} -> 200 OK (Entity: {detail['entity_id']}, Risk: {detail['risk_score']})")

    # 5. Test Entities API
    response = client.get("/api/entities")
    assert response.status_code == 200, f"/api/entities failed: {response.status_code}"
    entities_data = response.json()
    assert "entities" in entities_data, "Missing entities key"
    entities = entities_data["entities"]
    print(f"[OK] GET /api/entities -> 200 OK (Retrieved {len(entities)} entity profiles, Total: {entities_data['total']})")

    # 6. Test Entity Detail API
    if entities:
        eid = entities[0]["entity_id"]
        response = client.get(f"/api/entities/{eid}")
        assert response.status_code == 200, f"/api/entities/{eid} failed: {response.status_code}"
        edetail = response.json()
        assert "profile" in edetail, "Missing profile"
        print(f"[OK] GET /api/entities/{eid} -> 200 OK (Events: {len(edetail['recent_events'])}, Past Alerts: {len(edetail['past_alerts'])})")

    # 7. Test Metrics API
    response = client.get("/api/metrics")
    assert response.status_code == 200, f"/api/metrics failed: {response.status_code}"
    metrics = response.json()
    assert "pr_auc" in metrics, "Missing pr_auc in metrics"
    print(f"[OK] GET /api/metrics -> 200 OK (PR-AUC: {metrics['pr_auc']})")

    # 8. Test Simulate Attack API & GET /api/alerts after simulation
    res_sim = client.post("/api/simulate-attack?type=brute_force")
    assert res_sim.status_code == 200, f"/api/simulate-attack failed: {res_sim.status_code}"
    res_post_sim = client.get("/api/alerts?page=1&limit=10")
    assert res_post_sim.status_code == 200, f"/api/alerts failed after simulation: {res_post_sim.status_code}"
    print("[OK] POST /api/simulate-attack & GET /api/alerts after simulation -> 200 OK")

    # 9. Test CSV Export API
    res_export = client.get("/api/alerts/export?top_pct=1.0&min_risk=0.0&attack_type=all")
    assert res_export.status_code == 200, f"/api/alerts/export failed: {res_export.status_code}"
    assert "text/csv" in res_export.headers.get("content-type", ""), "Invalid export Content-Type"
    print("[OK] GET /api/alerts/export -> 200 OK (CSV Download)")

    print("\n[OK] ALL DASHBOARD ENDPOINTS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()
