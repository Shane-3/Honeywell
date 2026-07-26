# Data Generation Assumptions & Attack Taxonomy

This document describes the design decisions, behavioral assumptions, and attack injection
logic used by `generate_data.py`. It is a graded deliverable per the Honeywell brief.

---

## 1. Entity Population Design

### 1.1 Entity Types & Distribution
| Type | Fraction | ID Format | Rationale |
|---|---|---|---|
| `user` | 60% | `user_0001` | Human employees — largest population, most varied behavior |
| `service_account` | 20% | `svc_0001` | Automated processes — predictable schedules, narrow resource scope |
| `edge_device` | 20% | `device_0001` | IoT/OT endpoints — periodic telemetry, firmware updates |

### 1.2 Per-Entity Habitual Profile
Every entity is created with a **stable behavioral baseline** — normal events are sampled
from this per-entity profile, not from a global distribution. This ensures that "normal"
is entity-relative (matching the brief's framing).

Each profile includes:
- **Typical login hours**: Gaussian distribution centered on a business-hours peak (users),
  near-uniform with slight business-hours bias (service accounts), or uniform with slight
  night dip (edge devices).
- **Home geo-locations**: 1–2 typical cities drawn from a pool of 20 global cities.
- **Home IP prefixes**: One /24 prefix per home geo (synthetic, not real).
- **Typical resources**: 2–7 resources sampled from the entity-type-specific resource pool,
  with Dirichlet-distributed access frequency weights.
- **Primary auth method**: Weighted by entity type — users favor password/biometric,
  service accounts favor token/certificate, edge devices favor certificate.
- **Session duration**: Normal distribution with entity-specific mean (300–3600s for users,
  5–600s for service accounts, 10–300s for edge devices) and proportional std dev.
- **Device fingerprints**: 1–2 known `{os_version}|{mac_address}|{protocol}` strings.
- **Command vocabulary**: 3–8 typical commands for privileged sessions.
- **Sessions per day**: Poisson-distributed (λ=4 for users, λ=15 for service accounts,
  λ=8 for edge devices).

### 1.3 Noise in Normal Traffic
Normal events include **deliberate realistic noise** (FR-1.7):
- 5% chance of accessing a resource outside the entity's typical set (benign exploration)
- Occasional variation in auth method (6% non-primary)
- Session duration varies per Gaussian noise around the entity's mean
- This prevents the anomaly/normal boundary from being trivially separable

---

## 2. Additional Schema Field: `auth_result`

> **Design decision**: The brief's schema does not explicitly list `auth_result`, but brute
> force and credential stuffing attacks are defined entirely in terms of **failed authentication
> attempts**. Without `auth_result`, these attack types cannot be meaningfully simulated or
> detected. Therefore, `auth_result` (enum: `success` | `fail`) is included as a necessary
> internal field — this is not scope creep but a logical requirement of the attack taxonomy.

---

## 3. Attack Taxonomy — Injection Logic

All attacks are injected at a **controlled, low rate** (configurable, default 2% of total
sessions) with the exact count and percentage logged for reproducibility (FR-1.4).

### 3.1 Brute Force
- **Pattern**: 5–20 rapid failed-auth attempts from a single source IP in a 1–5 minute window
- **Auth method**: Always `password`
- **Auth result**: All `fail`, with 50% chance of a final `success` (compromised credential)
- **Geo/IP**: Random (attacker IP, not home IP)
- **Session duration**: 1–5 seconds (failed attempts are brief)

### 3.2 Impossible Travel
- **Pattern**: Same `entity_id` logs in from two locations >3,000 km apart within 10–45 minutes
- **Geo selection**: Two cities from the global pool guaranteed to be >3,000 km apart
- **Auth result**: Both `success` (the attacker has valid credentials)
- **Primarily targets**: User entities (the most realistic travel scenario)

### 3.3 Credential Stuffing
- **Pattern**: 5–20 different `entity_id`s attempted from 1–3 `source_ip`s in a short window
- **Auth result**: 85% `fail`, 15% `success` (automated credential list attack)
- **Auth method**: Always `password`
- **Resource**: Always `/api/login` (login endpoint)
- **Cross-entity**: This is the only attack that targets multiple entities in a single incident

### 3.4 Lateral Movement
- **Pattern**: Compromised entity accesses 4–8 resources **outside** its typical resource set
  in a short time window (minutes to an hour)
- **Commands**: Suspicious commands (nmap, whoami, shadow_read, nc) not in the entity's
  typical vocabulary
- **Geo/IP**: From the entity's home location (already inside the network)

### 3.5 Device Spoofing
- **Pattern**: An entity appears with a device fingerprint that doesn't match any of its
  known fingerprints (different OS, different MAC, different protocol)
- **Auth result**: `success` (the spoofed device has valid credentials)
- **Generates**: Single event per incident

### 3.6 Low-and-Slow Exfiltration
- **Pattern**: Gradual, small off-hours (22:00–05:00) resource accesses building up over
  7–21 days, targeting sensitive resources (finance, audit-log, secrets-vault, backup)
- **Commands**: Data exfiltration tools (scp, tar, rsync, wget)
- **Session duration**: Short (5–120 seconds)
- **Multi-timescale**: Deliberately spans many days — a single-session model cannot detect
  this; requires a multi-day rolling window approach

### 3.7 Insider Drift (Edge Case)
- **Pattern**: Legitimate user entity slowly expanding its resource footprint over 10–30 days
- **Mechanism**: Probability of accessing new (non-typical) resources increases linearly with
  drift progress (from 30% to 80% over the drift period)
- **Work hours**: Mostly during business hours (this is a legitimate user, not an attacker)
- **Auth method**: Occasionally shifts (10% × drift_progress chance of using a new method)
- **Ambiguity by design**: This is expected to be confused with lateral movement or normal
  behavior in classification — the report will discuss this honestly as a false-positive
  tuning challenge, not hide it

---

## 4. Volume & Determinism (FR-1.6)

| Parameter | Default | CLI Flag |
|---|---|---|
| Entities | 500 | `--n-entities` |
| Days | 45 | `--n-days` |
| Attack rate | 2% | `--attack-rate` |
| Random seed | 42 | `--seed` |

Expected output: ~50,000–120,000 total events depending on entity activity rates.

---

## 5. Privacy (NFR-6)

All identifiers (IPs, MAC addresses, entity IDs) are synthetic. IP addresses and MAC
addresses are generated via `faker`. No real employee names, IPs, or organizational data
are used anywhere in the generated dataset.

---

## 6. Ground Truth Handling (FR-1.5)

The `label` column contains ground truth for every event. It is:
- **Retained** in the CSV for training and offline evaluation
- **Never used as an input feature** by the detection model or dashboard at inference time
- The dashboard's "demo mode" ground-truth display is explicitly labeled as such
