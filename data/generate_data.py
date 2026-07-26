"""
Synthetic Access Log Generator for Behavioral Anomaly Detection
================================================================
Generates realistic access/connection events for a population of entities
(users, service accounts, edge devices), each with a stable habitual profile.
Injects 7 attack patterns + 1 edge case (insider drift) at configurable rates.

Usage:
    python data/generate_data.py --seed 42 --n-entities 500 --n-days 45 --attack-rate 0.02
"""

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from faker import Faker

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TYPE_DISTRIBUTION = {
    "user": 0.60,
    "service_account": 0.20,
    "edge_device": 0.20,
}

AUTH_METHODS = ["password", "token", "certificate", "biometric"]

# Weighted auth method distributions per entity type
AUTH_METHOD_WEIGHTS = {
    "user": [0.50, 0.25, 0.10, 0.15],
    "service_account": [0.05, 0.60, 0.30, 0.05],
    "edge_device": [0.05, 0.20, 0.70, 0.05],
}

# Resources per entity type (pool to sample from)
RESOURCE_POOLS = {
    "user": [
        "/api/documents", "/api/email", "/api/calendar", "/api/reports",
        "/api/analytics", "/api/crm", "/api/hr-portal", "/api/finance",
        "/api/project-mgmt", "/api/wiki", "/api/chat", "/api/video-conf",
        "/api/code-repo", "/api/ci-cd", "/api/cloud-console", "/api/vpn",
        "/api/file-share", "/api/admin-panel", "/api/audit-log", "/api/settings",
    ],
    "service_account": [
        "/api/internal/db-query", "/api/internal/batch-job", "/api/internal/etl",
        "/api/internal/monitoring", "/api/internal/deploy", "/api/internal/config",
        "/api/internal/secrets-vault", "/api/internal/log-ingestion",
        "/api/internal/backup", "/api/internal/health-check",
        "/api/internal/cache-invalidate", "/api/internal/queue-worker",
    ],
    "edge_device": [
        "/iot/telemetry", "/iot/firmware-update", "/iot/config-sync",
        "/iot/status-report", "/iot/alert-push", "/iot/command-recv",
        "/iot/data-upload", "/iot/heartbeat", "/iot/diagnostics",
        "/iot/edge-compute", "/iot/sensor-calibrate", "/iot/log-sync",
    ],
}

# Command vocabularies for privileged sessions
COMMAND_POOLS = {
    "user": [
        "ls", "cd", "cat", "grep", "find", "vim", "nano", "ssh", "scp",
        "docker ps", "kubectl get pods", "git pull", "git push", "make",
        "python run.py", "npm start", "curl", "wget", "tail -f",
    ],
    "service_account": [
        "SELECT * FROM", "INSERT INTO", "UPDATE", "DELETE FROM",
        "CREATE INDEX", "pg_dump", "redis-cli", "celery worker",
        "python etl.py", "spark-submit", "airflow trigger_dag",
    ],
    "edge_device": [
        "sensor_read", "actuator_set", "config_reload", "firmware_check",
        "diagnostic_run", "reboot", "log_rotate", "calibrate",
    ],
}

# Geo locations (city, country, lat, lon) for synthetic IP mapping
GEO_LOCATIONS = [
    ("New York", "US", 40.7128, -74.0060),
    ("San Francisco", "US", 37.7749, -122.4194),
    ("Chicago", "US", 41.8781, -87.6298),
    ("London", "UK", 51.5074, -0.1278),
    ("Berlin", "DE", 52.5200, 13.4050),
    ("Paris", "FR", 48.8566, 2.3522),
    ("Tokyo", "JP", 35.6762, 139.6503),
    ("Sydney", "AU", -33.8688, 151.2093),
    ("Mumbai", "IN", 19.0760, 72.8777),
    ("Singapore", "SG", 1.3521, 103.8198),
    ("Toronto", "CA", 43.6532, -79.3832),
    ("Dubai", "AE", 25.2048, 55.2708),
    ("São Paulo", "BR", -23.5505, -46.6333),
    ("Seoul", "KR", 37.5665, 126.9780),
    ("Amsterdam", "NL", 52.3676, 4.9041),
    ("Stockholm", "SE", 59.3293, 18.0686),
    ("Moscow", "RU", 55.7558, 37.6173),
    ("Cape Town", "ZA", -33.9249, 18.4241),
    ("Mexico City", "MX", 19.4326, -99.1332),
    ("Bangkok", "TH", 13.7563, 100.5018),
]

# OS/firmware versions for device fingerprints
OS_VERSIONS = [
    "Windows 11 23H2", "Windows 10 22H2", "macOS 14.2", "macOS 13.6",
    "Ubuntu 22.04", "Ubuntu 20.04", "RHEL 9.3", "CentOS 8",
    "iOS 17.2", "Android 14", "FreeRTOS 10.5", "Zephyr 3.5",
    "EdgeOS 2.1", "IndustrialOS 4.0", "PLC-Firmware 7.2",
]

PROTOCOLS = ["HTTPS", "SSH", "MQTT", "CoAP", "TLS-1.3", "gRPC", "WebSocket"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ─────────────────────────────────────────────────────────────────────────────
# Entity Profile Generation
# ─────────────────────────────────────────────────────────────────────────────

class EntityProfile:
    """A synthetic entity with stable behavioral habits."""

    def __init__(self, entity_id, entity_type, rng, fake):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.rng = rng

        # Home geo: 1-2 typical locations
        n_home_geos = rng.choice([1, 1, 1, 2])  # most entities have 1 location
        geo_indices = rng.choice(len(GEO_LOCATIONS), size=n_home_geos, replace=False)
        self.home_geos = [GEO_LOCATIONS[i] for i in geo_indices]
        self.home_geo_names = [f"{g[0]}, {g[1]}" for g in self.home_geos]

        # Home IP prefixes (1 per geo)
        self.home_ips = []
        for _ in self.home_geos:
            prefix = f"{rng.integers(10, 200)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}"
            self.home_ips.append(prefix)

        # Typical login hours: pick a center hour and spread
        if entity_type == "user":
            center_hour = rng.choice([9, 10, 11, 14, 15])  # business hours
            self.hour_probs = self._make_hour_distribution(center_hour, spread=2.5)
        elif entity_type == "service_account":
            # Service accounts run around the clock, slight peak during business
            self.hour_probs = np.ones(24) / 24
            self.hour_probs[8:18] *= 1.5
            self.hour_probs /= self.hour_probs.sum()
        else:  # edge_device
            # IoT devices: nearly uniform, slight dip at night
            self.hour_probs = np.ones(24) / 24
            self.hour_probs[2:5] *= 0.7
            self.hour_probs /= self.hour_probs.sum()

        # Typical resources: subset of the pool
        pool = RESOURCE_POOLS[entity_type]
        n_typical = rng.integers(2, min(7, len(pool)) + 1)
        self.typical_resources = list(rng.choice(pool, size=n_typical, replace=False))
        # Resource frequency weights (some used more than others)
        raw_weights = rng.dirichlet(np.ones(n_typical) * 2)
        self.resource_weights = raw_weights / raw_weights.sum()

        # Auth method: pick a primary with occasional alternatives
        weights = AUTH_METHOD_WEIGHTS[entity_type]
        self.primary_auth = rng.choice(AUTH_METHODS, p=weights)
        self.auth_probs = {m: 0.02 for m in AUTH_METHODS}
        self.auth_probs[self.primary_auth] = 0.94  # heavily favor primary

        # Session duration: normal distribution params (seconds)
        if entity_type == "user":
            self.duration_mean = rng.integers(300, 3600)
            self.duration_std = max(60, self.duration_mean * 0.3)
        elif entity_type == "service_account":
            self.duration_mean = rng.integers(5, 600)
            self.duration_std = max(5, self.duration_mean * 0.4)
        else:
            self.duration_mean = rng.integers(10, 300)
            self.duration_std = max(5, self.duration_mean * 0.25)

        # Device fingerprint(s): 1-2 known devices
        n_devices = rng.choice([1, 1, 2])
        self.known_fingerprints = []
        for _ in range(n_devices):
            os_ver = rng.choice(OS_VERSIONS)
            mac = fake.mac_address()
            proto = rng.choice(PROTOCOLS)
            self.known_fingerprints.append(f"{os_ver}|{mac}|{proto}")

        # Command vocabulary (for privileged sessions)
        cmd_pool = COMMAND_POOLS[entity_type]
        n_cmds = rng.integers(3, min(8, len(cmd_pool)) + 1)
        self.typical_commands = list(rng.choice(cmd_pool, size=n_cmds, replace=False))

        # Whether this entity has privileged sessions
        if entity_type == "user":
            self.privileged_prob = 0.15  # 15% of user sessions are privileged
        elif entity_type == "service_account":
            self.privileged_prob = 0.70
        else:
            self.privileged_prob = 0.30

        # Sessions per day: how active is this entity
        if entity_type == "user":
            self.sessions_per_day = max(1, rng.poisson(4))
        elif entity_type == "service_account":
            self.sessions_per_day = max(1, rng.poisson(15))
        else:
            self.sessions_per_day = max(1, rng.poisson(8))

    def _make_hour_distribution(self, center, spread):
        """Gaussian-like distribution over 24 hours centered at `center`."""
        hours = np.arange(24)
        probs = np.exp(-0.5 * ((hours - center) / spread) ** 2)
        # Add small baseline for occasional off-hours activity
        probs += 0.01
        return probs / probs.sum()

    def sample_normal_event(self, day_offset, start_date, rng):
        """Generate a single normal event from this entity's profile."""
        # Timestamp
        hour = rng.choice(24, p=self.hour_probs)
        minute = rng.integers(0, 60)
        second = rng.integers(0, 60)
        ts = start_date + timedelta(days=int(day_offset), hours=int(hour),
                                     minutes=int(minute), seconds=int(second))

        # Geo & IP (from home locations)
        geo_idx = rng.integers(0, len(self.home_geos))
        geo = self.home_geos[geo_idx]
        geo_name = self.home_geo_names[geo_idx]
        ip_prefix = self.home_ips[geo_idx]
        source_ip = f"{ip_prefix}.{rng.integers(1, 255)}"

        # Resource
        res_idx = rng.choice(len(self.typical_resources), p=self.resource_weights)
        resource = self.typical_resources[res_idx]

        # Occasional noise: 5% chance of accessing a slightly unusual resource
        if rng.random() < 0.05:
            pool = RESOURCE_POOLS[self.entity_type]
            resource = rng.choice(pool)

        # Auth method
        auth_probs_list = [self.auth_probs.get(m, 0.02) for m in AUTH_METHODS]
        auth_probs_arr = np.array(auth_probs_list)
        auth_probs_arr /= auth_probs_arr.sum()
        auth_method = rng.choice(AUTH_METHODS, p=auth_probs_arr)

        # Session duration
        duration = max(1, int(rng.normal(self.duration_mean, self.duration_std)))

        # Device fingerprint
        fingerprint = rng.choice(self.known_fingerprints)

        # Commands (for privileged sessions)
        commands = []
        if rng.random() < self.privileged_prob:
            n_cmds = rng.integers(1, min(5, len(self.typical_commands)) + 1)
            commands = list(rng.choice(self.typical_commands, size=n_cmds, replace=False))

        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": source_ip,
            "geo_location": geo_name,
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_result": "success",
            "session_duration": duration,
            "command_sequence": json.dumps(commands),
            "device_fingerprint": fingerprint,
            "label": "normal",
            "session_id": str(uuid.uuid4())[:12],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Attack Injection Functions
# ─────────────────────────────────────────────────────────────────────────────

def inject_brute_force(entity, day_offset, start_date, rng, fake):
    """
    Brute force: 5-20 rapid failed-auth attempts from one source in a short
    window (1-5 minutes), followed by one optional success.
    """
    events = []
    n_attempts = rng.integers(5, 21)
    hour = rng.integers(0, 24)
    minute = rng.integers(0, 55)
    base_ts = start_date + timedelta(days=int(day_offset), hours=int(hour),
                                      minutes=int(minute))
    source_ip = f"{rng.integers(50, 220)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
    geo_idx = rng.integers(0, len(GEO_LOCATIONS))
    geo = GEO_LOCATIONS[geo_idx]
    geo_name = f"{geo[0]}, {geo[1]}"

    for i in range(n_attempts):
        ts = base_ts + timedelta(seconds=int(rng.integers(5, 30) * i))
        events.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": source_ip,
            "geo_location": geo_name,
            "resource_accessed": rng.choice(entity.typical_resources) if entity.typical_resources else "/api/login",
            "auth_method": "password",
            "auth_result": "fail",
            "session_duration": rng.integers(1, 5),
            "command_sequence": json.dumps([]),
            "device_fingerprint": rng.choice(entity.known_fingerprints),
            "label": "brute_force",
            "session_id": str(uuid.uuid4())[:12],
        })

    # Optional final success (50% chance)
    if rng.random() < 0.5:
        ts = base_ts + timedelta(seconds=int(rng.integers(5, 30) * n_attempts))
        events.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": source_ip,
            "geo_location": geo_name,
            "resource_accessed": rng.choice(entity.typical_resources) if entity.typical_resources else "/api/login",
            "auth_method": "password",
            "auth_result": "success",
            "session_duration": rng.integers(30, 300),
            "command_sequence": json.dumps([]),
            "device_fingerprint": rng.choice(entity.known_fingerprints),
            "label": "brute_force",
            "session_id": str(uuid.uuid4())[:12],
        })

    return events


def inject_impossible_travel(entity, day_offset, start_date, rng, fake):
    """
    Impossible travel: same entity logs in from two distant locations
    within an implausibly short time gap (< 1 hour, > 3000km apart).
    """
    events = []
    hour = rng.integers(6, 22)
    base_ts = start_date + timedelta(days=int(day_offset), hours=int(hour),
                                      minutes=int(rng.integers(0, 30)))

    # Pick two distant geos
    geo1_idx = rng.integers(0, len(GEO_LOCATIONS))
    attempts = 0
    while attempts < 50:
        geo2_idx = rng.integers(0, len(GEO_LOCATIONS))
        if geo2_idx != geo1_idx:
            dist = haversine_km(
                GEO_LOCATIONS[geo1_idx][2], GEO_LOCATIONS[geo1_idx][3],
                GEO_LOCATIONS[geo2_idx][2], GEO_LOCATIONS[geo2_idx][3],
            )
            if dist > 3000:
                break
        attempts += 1

    geo1 = GEO_LOCATIONS[geo1_idx]
    geo2 = GEO_LOCATIONS[geo2_idx]

    # First login from geo1
    ip1 = f"{rng.integers(10, 200)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
    events.append({
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "timestamp": base_ts.isoformat(),
        "source_ip": ip1,
        "geo_location": f"{geo1[0]}, {geo1[1]}",
        "resource_accessed": rng.choice(entity.typical_resources) if entity.typical_resources else "/api/login",
        "auth_method": rng.choice(AUTH_METHODS),
        "auth_result": "success",
        "session_duration": rng.integers(60, 600),
        "command_sequence": json.dumps([]),
        "device_fingerprint": rng.choice(entity.known_fingerprints),
        "label": "impossible_travel",
        "session_id": str(uuid.uuid4())[:12],
    })

    # Second login from geo2, 10-45 minutes later
    gap_minutes = rng.integers(10, 46)
    ts2 = base_ts + timedelta(minutes=int(gap_minutes))
    ip2 = f"{rng.integers(10, 200)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
    events.append({
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "timestamp": ts2.isoformat(),
        "source_ip": ip2,
        "geo_location": f"{geo2[0]}, {geo2[1]}",
        "resource_accessed": rng.choice(entity.typical_resources) if entity.typical_resources else "/api/login",
        "auth_method": rng.choice(AUTH_METHODS),
        "auth_result": "success",
        "session_duration": rng.integers(60, 600),
        "command_sequence": json.dumps([]),
        "device_fingerprint": rng.choice(entity.known_fingerprints),
        "label": "impossible_travel",
        "session_id": str(uuid.uuid4())[:12],
    })

    return events


def inject_credential_stuffing(entities, day_offset, start_date, rng, fake):
    """
    Credential stuffing: many entity_ids attempted from few source_ips,
    high failure rate. This is a cross-entity attack, so it takes a list
    of entities and picks targets.
    """
    events = []
    n_targets = rng.integers(5, 20)
    n_source_ips = rng.integers(1, 4)

    source_ips = [
        f"{rng.integers(50, 220)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
        for _ in range(n_source_ips)
    ]
    geo_idx = rng.integers(0, len(GEO_LOCATIONS))
    geo = GEO_LOCATIONS[geo_idx]
    geo_name = f"{geo[0]}, {geo[1]}"

    target_entities = rng.choice(entities, size=min(n_targets, len(entities)), replace=False)
    hour = rng.integers(0, 24)
    base_ts = start_date + timedelta(days=int(day_offset), hours=int(hour),
                                      minutes=int(rng.integers(0, 50)))

    for i, entity in enumerate(target_entities):
        ts = base_ts + timedelta(seconds=int(rng.integers(1, 15) * i))
        # 85% fail, 15% success
        auth_result = "fail" if rng.random() < 0.85 else "success"
        events.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": rng.choice(source_ips),
            "geo_location": geo_name,
            "resource_accessed": "/api/login",
            "auth_method": "password",
            "auth_result": auth_result,
            "session_duration": rng.integers(1, 10),
            "command_sequence": json.dumps([]),
            "device_fingerprint": rng.choice(entity.known_fingerprints),
            "label": "credential_stuffing",
            "session_id": str(uuid.uuid4())[:12],
        })

    return events


def inject_lateral_movement(entity, day_offset, start_date, rng, fake):
    """
    Lateral movement: entity accesses an unusual breadth of resources
    never touched before, in a short time window.
    """
    events = []
    # Pick 4-8 resources NOT in the entity's typical set
    all_resources = []
    for pool in RESOURCE_POOLS.values():
        all_resources.extend(pool)
    unusual_resources = [r for r in all_resources if r not in entity.typical_resources]
    n_access = min(rng.integers(4, 9), len(unusual_resources))
    targets = list(rng.choice(unusual_resources, size=n_access, replace=False))

    hour = rng.integers(0, 24)
    base_ts = start_date + timedelta(days=int(day_offset), hours=int(hour),
                                      minutes=int(rng.integers(0, 50)))

    for i, resource in enumerate(targets):
        ts = base_ts + timedelta(minutes=int(rng.integers(1, 15) * i))
        events.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": f"{entity.home_ips[0]}.{rng.integers(1, 255)}",
            "geo_location": entity.home_geo_names[0],
            "resource_accessed": resource,
            "auth_method": entity.primary_auth,
            "auth_result": "success",
            "session_duration": rng.integers(30, 600),
            "command_sequence": json.dumps(
                list(rng.choice(["ls", "cat", "find", "wget", "curl", "scp",
                                  "tar", "nc", "nmap", "whoami", "id",
                                  "passwd", "shadow_read"], size=rng.integers(2, 5)))
            ),
            "device_fingerprint": rng.choice(entity.known_fingerprints),
            "label": "lateral_movement",
            "session_id": str(uuid.uuid4())[:12],
        })

    return events


def inject_device_spoofing(entity, day_offset, start_date, rng, fake):
    """
    Device spoofing: same entity appears with a completely mismatched
    device fingerprint (different OS, MAC, protocol vs. known history).
    """
    events = []
    hour = rng.integers(0, 24)
    ts = start_date + timedelta(days=int(day_offset), hours=int(hour),
                                 minutes=int(rng.integers(0, 60)))

    # Generate a fake fingerprint that doesn't match known ones
    spoofed_os = rng.choice([v for v in OS_VERSIONS
                              if not any(v in fp for fp in entity.known_fingerprints)])
    spoofed_mac = fake.mac_address()
    spoofed_proto = rng.choice(PROTOCOLS)
    spoofed_fp = f"{spoofed_os}|{spoofed_mac}|{spoofed_proto}"

    events.append({
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "timestamp": ts.isoformat(),
        "source_ip": f"{rng.integers(10, 200)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}",
        "geo_location": entity.home_geo_names[0],
        "resource_accessed": rng.choice(entity.typical_resources) if entity.typical_resources else "/api/login",
        "auth_method": entity.primary_auth,
        "auth_result": "success",
        "session_duration": rng.integers(60, 1800),
        "command_sequence": json.dumps([]),
        "device_fingerprint": spoofed_fp,
        "label": "device_spoofing",
        "session_id": str(uuid.uuid4())[:12],
    })

    return events


def inject_low_and_slow(entity, start_day, n_days, start_date, rng, fake):
    """
    Low-and-slow exfiltration: gradual, small, off-hours resource access
    building up over days/weeks. Generates events spread across many days.
    """
    events = []
    # Spread events over 7-21 days
    span = min(rng.integers(7, 22), n_days - start_day)
    n_events = rng.integers(span, span * 3)

    # Target sensitive resources
    sensitive = ["/api/finance", "/api/audit-log", "/api/admin-panel",
                 "/api/internal/secrets-vault", "/api/internal/db-query",
                 "/api/internal/backup", "/api/file-share", "/api/documents"]

    for i in range(n_events):
        day = start_day + int(i * span / n_events)
        # Off-hours: 22-05
        hour = rng.choice([22, 23, 0, 1, 2, 3, 4, 5])
        minute = rng.integers(0, 60)
        ts = start_date + timedelta(days=int(day), hours=int(hour),
                                     minutes=int(minute))

        events.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": f"{entity.home_ips[0]}.{rng.integers(1, 255)}",
            "geo_location": entity.home_geo_names[0],
            "resource_accessed": rng.choice(sensitive),
            "auth_method": entity.primary_auth,
            "auth_result": "success",
            "session_duration": rng.integers(5, 120),  # short sessions
            "command_sequence": json.dumps(
                list(rng.choice(["scp", "tar", "zip", "curl -O", "wget -q",
                                  "rsync", "cp"], size=rng.integers(1, 3)))
            ),
            "device_fingerprint": rng.choice(entity.known_fingerprints),
            "label": "low_and_slow_exfil",
            "session_id": str(uuid.uuid4())[:12],
        })

    return events


def inject_insider_drift(entity, start_day, n_days, start_date, rng, fake):
    """
    Insider drift: legitimate entity slowly expanding privilege/resource
    footprint over weeks. This is the ambiguous edge case — the entity is
    legitimate, but their behavior is gradually shifting.
    """
    events = []
    # Drift spans 10-30 days
    span = min(rng.integers(10, 31), n_days - start_day)
    n_events = rng.integers(span, span * 2)

    # Gradually introduce new resources from outside typical set
    all_resources = RESOURCE_POOLS[entity.entity_type]
    new_resources = [r for r in all_resources if r not in entity.typical_resources]
    if not new_resources:
        new_resources = ["/api/admin-panel", "/api/internal/secrets-vault"]

    for i in range(n_events):
        day = start_day + int(i * span / n_events)
        # Mostly during work hours (this is a legitimate user shifting behavior)
        hour = rng.choice(list(range(7, 20)))
        minute = rng.integers(0, 60)
        ts = start_date + timedelta(days=int(day), hours=int(hour),
                                     minutes=int(minute))

        # Gradually increase probability of accessing new resources
        drift_progress = i / max(n_events - 1, 1)  # 0 to 1
        if rng.random() < 0.3 + 0.5 * drift_progress:
            resource = rng.choice(new_resources)
        else:
            resource = rng.choice(entity.typical_resources)

        # Occasionally use a new auth method
        if rng.random() < 0.1 * drift_progress:
            auth_method = rng.choice(AUTH_METHODS)
        else:
            auth_method = entity.primary_auth

        events.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts.isoformat(),
            "source_ip": f"{entity.home_ips[0]}.{rng.integers(1, 255)}",
            "geo_location": entity.home_geo_names[0],
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_result": "success",
            "session_duration": rng.integers(60, 1800),
            "command_sequence": json.dumps(
                list(rng.choice(entity.typical_commands,
                                 size=rng.integers(1, min(4, len(entity.typical_commands)) + 1)))
                if rng.random() < entity.privileged_prob else []
            ),
            "device_fingerprint": rng.choice(entity.known_fingerprints),
            "label": "insider_drift",
            "session_id": str(uuid.uuid4())[:12],
        })

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(seed=42, n_entities=500, n_days=45, attack_rate=0.02):
    """
    Generate the full synthetic access log dataset.

    Args:
        seed: Random seed for reproducibility
        n_entities: Number of entities to create
        n_days: Number of days to simulate
        attack_rate: Overall fraction of sessions that are attacks (0.005 to 0.03)

    Returns:
        pd.DataFrame with all events
    """
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    start_date = datetime(2025, 6, 1, 0, 0, 0)

    print(f"[*] Generating {n_entities} entities across {n_days} days (seed={seed})...")

    # ── Step 1: Create entity population ──────────────────────────────────
    entities = []
    entity_type_counts = defaultdict(int)

    for i in range(n_entities):
        # Determine entity type
        r = rng.random()
        cumulative = 0
        for etype, prob in ENTITY_TYPE_DISTRIBUTION.items():
            cumulative += prob
            if r <= cumulative:
                entity_type = etype
                break

        entity_type_counts[entity_type] += 1
        count = entity_type_counts[entity_type]

        if entity_type == "user":
            eid = f"user_{count:04d}"
        elif entity_type == "service_account":
            eid = f"svc_{count:04d}"
        else:
            eid = f"device_{count:04d}"

        profile = EntityProfile(eid, entity_type, rng, fake)
        entities.append(profile)

    print(f"    Entity population: {dict(entity_type_counts)}")

    # ── Step 2: Generate normal traffic ───────────────────────────────────
    print("[*] Generating normal traffic...")
    all_events = []

    for entity in entities:
        for day in range(n_days):
            # Vary daily session count slightly
            n_sessions = max(1, rng.poisson(entity.sessions_per_day))
            for _ in range(n_sessions):
                event = entity.sample_normal_event(day, start_date, rng)
                all_events.append(event)

    total_normal = len(all_events)
    print(f"    Normal events: {total_normal}")

    # ── Step 3: Inject attacks ────────────────────────────────────────────
    print("[*] Injecting attack patterns...")
    attack_events = []
    attack_counts = defaultdict(int)

    # Calculate budget per attack type
    total_attack_budget = int(total_normal * attack_rate / (1 - attack_rate))
    # Distribute: brute force and credential stuffing generate more events per
    # incident, others fewer. Allocate by incident count, not event count.
    per_type_incidents = max(1, total_attack_budget // 50)  # rough incident count

    # Per-type rates (relative weights)
    type_weights = {
        "brute_force": 1.5,
        "impossible_travel": 1.0,
        "credential_stuffing": 1.0,
        "lateral_movement": 1.2,
        "device_spoofing": 0.8,
        "low_and_slow_exfil": 0.7,
        "insider_drift": 0.8,
    }
    total_weight = sum(type_weights.values())

    # Brute Force
    n_incidents = max(2, int(per_type_incidents * type_weights["brute_force"] / total_weight))
    for _ in range(n_incidents):
        entity = rng.choice(entities)
        day = rng.integers(0, n_days)
        evts = inject_brute_force(entity, day, start_date, rng, fake)
        attack_events.extend(evts)
        attack_counts["brute_force"] += len(evts)

    # Impossible Travel
    n_incidents = max(2, int(per_type_incidents * type_weights["impossible_travel"] / total_weight))
    user_entities = [e for e in entities if e.entity_type == "user"]
    for _ in range(n_incidents):
        entity = rng.choice(user_entities) if user_entities else rng.choice(entities)
        day = rng.integers(0, n_days)
        evts = inject_impossible_travel(entity, day, start_date, rng, fake)
        attack_events.extend(evts)
        attack_counts["impossible_travel"] += len(evts)

    # Credential Stuffing
    n_incidents = max(2, int(per_type_incidents * type_weights["credential_stuffing"] / total_weight))
    for _ in range(n_incidents):
        day = rng.integers(0, n_days)
        evts = inject_credential_stuffing(entities, day, start_date, rng, fake)
        attack_events.extend(evts)
        attack_counts["credential_stuffing"] += len(evts)

    # Lateral Movement
    n_incidents = max(2, int(per_type_incidents * type_weights["lateral_movement"] / total_weight))
    for _ in range(n_incidents):
        entity = rng.choice(entities)
        day = rng.integers(0, n_days)
        evts = inject_lateral_movement(entity, day, start_date, rng, fake)
        attack_events.extend(evts)
        attack_counts["lateral_movement"] += len(evts)

    # Device Spoofing
    n_incidents = max(2, int(per_type_incidents * type_weights["device_spoofing"] / total_weight))
    for _ in range(n_incidents):
        entity = rng.choice(entities)
        day = rng.integers(0, n_days)
        evts = inject_device_spoofing(entity, day, start_date, rng, fake)
        attack_events.extend(evts)
        attack_counts["device_spoofing"] += len(evts)

    # Low-and-Slow Exfiltration
    n_incidents = max(2, int(per_type_incidents * type_weights["low_and_slow_exfil"] / total_weight))
    for _ in range(n_incidents):
        entity = rng.choice(entities)
        start_day = rng.integers(0, max(1, n_days - 10))
        evts = inject_low_and_slow(entity, start_day, n_days, start_date, rng, fake)
        attack_events.extend(evts)
        attack_counts["low_and_slow_exfil"] += len(evts)

    # Insider Drift
    n_incidents = max(2, int(per_type_incidents * type_weights["insider_drift"] / total_weight))
    for _ in range(n_incidents):
        entity = rng.choice(user_entities) if user_entities else rng.choice(entities)
        start_day = rng.integers(0, max(1, n_days - 15))
        evts = inject_insider_drift(entity, start_day, n_days, start_date, rng, fake)
        attack_events.extend(evts)
        attack_counts["insider_drift"] += len(evts)

    all_events.extend(attack_events)
    total_attacks = len(attack_events)
    total_events = len(all_events)

    print(f"    Attack events: {total_attacks} ({total_attacks/total_events*100:.2f}%)")
    print(f"    Breakdown:")
    for atype, count in sorted(attack_counts.items()):
        print(f"      {atype:25s}: {count:5d} ({count/total_events*100:.3f}%)")

    # ── Step 4: Shuffle and create DataFrame ──────────────────────────────
    print("[*] Building DataFrame...")
    df = pd.DataFrame(all_events)

    # Sort by timestamp (realistic ordering)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert timestamp back to ISO string for CSV
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"[OK] Dataset generated: {len(df)} total events, {df['entity_id'].nunique()} entities")
    print(f"    Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"    Label distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"      {label:25s}: {count:6d} ({count/len(df)*100:.2f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic access logs for behavioral anomaly detection"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-entities", type=int, default=500, help="Number of entities (default: 500)")
    parser.add_argument("--n-days", type=int, default=45, help="Number of simulated days (default: 45)")
    parser.add_argument("--attack-rate", type=float, default=0.02,
                        help="Overall attack fraction (default: 0.02 = 2%%)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data/synthetic_logs.csv)")
    args = parser.parse_args()

    df = generate_dataset(
        seed=args.seed,
        n_entities=args.n_entities,
        n_days=args.n_days,
        attack_rate=args.attack_rate,
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "synthetic_logs.csv")

    df.to_csv(output_path, index=False)
    print(f"[OK] Saved to {output_path}")


if __name__ == "__main__":
    main()
