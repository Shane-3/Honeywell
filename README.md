# Aegis AI — AI Powered Behavioral Anomaly Detection for Cybersecurity

**Enterprise Cybersecurity SOC Platform**

An end-to-end ML system that models "normal" access and connection behavior for users and devices, detects intrusions or compromised-credential activity, classifies the type of anomaly, provides explainable risk scores, and equips SOC analysts with live attack simulation and table export capabilities through a premium dashboard.

---

## Key Dashboard Features

- **Ranked Alert Queue**: Sortable, filterable table display with risk pills, attack categories, cold-start/drift badges, and search capability.
- **⚡ Live Cyber Attack Simulation Engine**: Inject real-time attack patterns (*Brute Force*, *Impossible Travel*, *Credential Stuffing*, *Lateral Movement*) with a single click and inspect telemetry in a rich modal pop-up.
- **📊 Table Results CSV Export Engine**: Instantly export the exact filtered table alerts matching search criteria, risk levels, and alert budget thresholds to `soc_alerts_report.csv`.
- **Analyst Alert Budget Simulator**: Dynamically adjust top-N% alert capacity sliders (0.1% to 10.0%) to match daily SOC triage bandwidth with live precision estimates.
- **Explainable Anomaly Attribution**: Human-readable explanations and top feature attribution vectors for every security incident.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Option A: Launch Dashboard Instantly (using pre-trained data & models)
python run_all.py --serve-only

# Option B: Re-run Full Pipeline (generate 160k logs → train → evaluate → launch)
python run_all.py --server

# The dashboard will open at http://localhost:8000
```

## Individual Steps

```bash
# Generate synthetic access logs
python data/generate_data.py --seed 42 --n-entities 500 --n-days 45

# Train all models and produce alerts
python models/train.py

# Run evaluation and generate report figures
python evaluation/evaluate.py

# Launch the Honeywell Sentinel AI dashboard
python dashboard/app.py
```

---

## Project Structure

```
honeywell-behavioral-anomaly/
├── data/
│   ├── generate_data.py          # Synthetic data generator
│   ├── data_assumptions.md       # Documented behavioral assumptions
│   └── synthetic_logs.csv        # Generated output (after running generator)
├── models/
│   ├── baseline_profile.py       # Per-entity statistical profiling + scoring
│   ├── sequence_model.py         # Markov/sliding-window sequence model
│   ├── classifier.py             # RandomForest + rule-based anomaly classifier
│   └── train.py                  # Training orchestrator
├── explain/
│   └── attribution.py            # Feature attribution + explanation generation
├── evaluation/
│   └── evaluate.py               # Metrics, PR curves, confusion matrix
├── dashboard/
│   ├── app.py                    # FastAPI backend (with simulation & CSV export API)
│   ├── static/
│   │   ├── css/styles.css        # Premium dark-mode glassmorphic design system
│   │   └── js/app.js             # SPA logic, attack simulation modal, charts
│   └── templates/
│       └── index.html            # Dashboard HTML shell
├── reports/
│   ├── report.md                 # Full project report
│   └── figures/                  # Generated evaluation charts
├── requirements.txt
├── run_all.py                    # One-command full pipeline orchestrator
└── README.md
```

---

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Data Generator  │────▶│  Training Pipeline│────▶│   Alerts CSV     │
│  (synthetic logs)│     │  (baseline +      │     │   (scored +      │
│                  │     │   sequence +      │     │    classified +  │
│                  │     │   classifier)     │     │    explained)    │
└─────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Evaluation      │◀─────────────┤
                        │  (metrics +      │              │
                        │   figures)       │              ▼
                        └──────────────────┘     ┌──────────────────┐
                                                 │ Honeywell        │
                                                 │ Sentinel AI      │
                                                 │ Dashboard        │
                                                 └──────────────────┘
```

---

## Attack Types Detected & Simulative Injection

| Attack Pattern | Description | Live Simulation Support |
|---|---|:---:|
| **Brute Force** | Rapid repeated failed-auth attempts | ✅ Supported |
| **Impossible Travel** | Logins from distant locations in implausible time | ✅ Supported |
| **Credential Stuffing** | Many entities, few IPs, high failure rate | ✅ Supported |
| **Lateral Movement** | Unusual breadth of new resource access | ✅ Supported |
| **Device Spoofing** | Device with mismatched fingerprint | Active Baseline |
| **Low-and-Slow Exfiltration** | Gradual off-hours access over days | Active Baseline |
| **Insider Drift** | Slow privilege/resource footprint expansion | Active Baseline |

---

## Tech Stack

- **Python 3.11** — end-to-end ML & backend
- **pandas / numpy / faker** — data manipulation & synthetic log generation
- **scikit-learn** — baseline profiling + classification
- **FastAPI + Uvicorn** — dashboard REST API backend
- **Vanilla HTML/CSS/JS + Chart.js** — glassmorphic dashboard frontend
- **matplotlib / seaborn** — evaluation visualizations

---

## License

Built for Honeywell hackathon evaluation.
