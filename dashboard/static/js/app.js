// Analyst Dashboard Single-Page Application Logic

let metricsSummary = null;

// Alert Queue Pagination State
let currentPage = 1;
const pageSize = 20;
let totalAlertsCount = 0;

// Entity Profiles Pagination State
let entityCurrentPage = 1;
const entityPageSize = 20;
let totalEntitiesCount = 0;

document.addEventListener('DOMContentLoaded', () => {
  // Navigation Routing
  const navLinks = document.querySelectorAll('.nav-link');
  const viewSections = document.querySelectorAll('.view-section');
  const viewTitle = document.getElementById('view-title');
  const viewSub = document.getElementById('view-sub');

  const viewHeaders = {
    'alerts-view': { title: 'Ranked Alert Queue', sub: 'Real-time AI behavioral anomaly detection feed' },
    'entities-view': { title: 'Entity Profiles & Baselines', sub: 'Behavioral footprints and historical risk posture' },
    'metrics-view': { title: 'Model Evaluation & System Metrics', sub: 'Imbalance-aware metrics, PR curves, and confusion matrix' }
  };

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetView = link.getAttribute('data-view');

      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      viewSections.forEach(sec => {
        if (sec.id === targetView) {
          sec.classList.add('active');
        } else {
          sec.classList.remove('active');
        }
      });

      if (viewHeaders[targetView]) {
        viewTitle.textContent = viewHeaders[targetView].title;
        viewSub.textContent = viewHeaders[targetView].sub;
      }

      if (targetView === 'entities-view') loadEntities();
      if (targetView === 'metrics-view') loadMetricsData();
    });
  });

  // Load Initial Data for all views
  loadStats();
  loadMetricsData();
  loadAlerts();
  loadEntities();

  // Alerts Filters & Event Listeners
  const searchInput = document.getElementById('search-input');
  const typeSelect = document.getElementById('type-select');
  const riskSelect = document.getElementById('risk-select');
  const budgetSlider = document.getElementById('alert-budget-slider');
  const budgetPctVal = document.getElementById('budget-pct-val');

  let debounceTimer;
  function triggerSearch() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      currentPage = 1;
      loadAlerts();
    }, 200);
  }

  if (searchInput) searchInput.addEventListener('input', triggerSearch);
  if (typeSelect) typeSelect.addEventListener('change', triggerSearch);
  if (riskSelect) riskSelect.addEventListener('change', triggerSearch);

  if (budgetSlider) {
    budgetSlider.addEventListener('input', (e) => {
      const pct = parseFloat(e.target.value).toFixed(1);
      budgetPctVal.textContent = `${pct}%`;
      updateBudgetPrecisionDisplay(pct);
      triggerSearch();
    });
  }

  // Alerts Pagination Controls
  const btnPrev = document.getElementById('btn-prev-page');
  const btnNext = document.getElementById('btn-next-page');

  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        loadAlerts();
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      const totalPages = Math.ceil(totalAlertsCount / pageSize) || 1;
      if (currentPage < totalPages) {
        currentPage++;
        loadAlerts();
      }
    });
  }

  // Entity Profiles Filters & Event Listeners
  const entitySearchInput = document.getElementById('entity-search-input');
  const entityTypeSelect = document.getElementById('entity-type-select');

  let entityDebounceTimer;
  function triggerEntitySearch() {
    clearTimeout(entityDebounceTimer);
    entityDebounceTimer = setTimeout(() => {
      entityCurrentPage = 1;
      loadEntities();
    }, 200);
  }

  if (entitySearchInput) entitySearchInput.addEventListener('input', triggerEntitySearch);
  if (entityTypeSelect) entityTypeSelect.addEventListener('change', triggerEntitySearch);

  // Action Buttons
  const btnExport = document.getElementById('btn-export-csv');
  const btnTableExport = document.getElementById('btn-export-table-csv');
  const btnSimulate = document.getElementById('btn-simulate-attack');

  const triggerExport = () => {
    const search = document.getElementById('search-input')?.value || '';
    const type = document.getElementById('type-select')?.value || 'all';
    const minRisk = document.getElementById('risk-select')?.value || '0.0';
    const topPct = document.getElementById('alert-budget-slider')?.value || '1.0';

    let exportUrl = `/api/alerts/export?top_pct=${topPct}&min_risk=${minRisk}&attack_type=${type}`;
    if (search) exportUrl += `&search=${encodeURIComponent(search)}`;

    window.location.href = exportUrl;
  };

  if (btnExport) btnExport.addEventListener('click', triggerExport);
  if (btnTableExport) btnTableExport.addEventListener('click', triggerExport);

  if (btnSimulate) {
    btnSimulate.addEventListener('click', () => {
      openSimulationModal();
    });
  }
  const btnEntityPrev = document.getElementById('btn-entity-prev-page');
  const btnEntityNext = document.getElementById('btn-entity-next-page');

  if (btnEntityPrev) {
    btnEntityPrev.addEventListener('click', () => {
      if (entityCurrentPage > 1) {
        entityCurrentPage--;
        loadEntities();
      }
    });
  }

  if (btnEntityNext) {
    btnEntityNext.addEventListener('click', () => {
      const totalPages = Math.ceil(totalEntitiesCount / entityPageSize) || 1;
      if (entityCurrentPage < totalPages) {
        entityCurrentPage++;
        loadEntities();
      }
    });
  }

  // Modal Close
  const modal = document.getElementById('detail-modal');
  const modalClose = document.getElementById('modal-close');
  if (modalClose) {
    modalClose.addEventListener('click', closeModal);
  }
  window.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });
});

function closeModal() {
  const modal = document.getElementById('detail-modal');
  if (modal) modal.style.display = 'none';
}

// Fetch Metrics Summary for Live Slider Calculations
async function loadMetricsData() {
  try {
    const res = await fetch('/api/metrics');
    metricsSummary = await res.json();
    updateBudgetPrecisionDisplay(1.0);
  } catch (err) {
    console.error('Failed to load metrics data:', err);
  }
}

// Live Alert Budget Precision Interpolator (FR-7.5)
function updateBudgetPrecisionDisplay(pct) {
  const lbl = document.getElementById('budget-precision-lbl');
  if (!lbl) return;

  pct = parseFloat(pct);
  let estPrecision = 31.16;

  if (metricsSummary && metricsSummary.alert_budget_precision) {
    const b = metricsSummary.alert_budget_precision;
    const p05 = (b["top_0.5%"] ? b["top_0.5%"].precision : 0.5896) * 100;
    const p10 = (b["top_1.0%"] ? b["top_1.0%"].precision : 0.3116) * 100;
    const p30 = (b["top_3.0%"] ? b["top_3.0%"].precision : 0.1169) * 100;

    if (pct <= 0.5) {
      estPrecision = p05 + (0.5 - pct) * 15;
    } else if (pct <= 1.0) {
      estPrecision = p05 - ((pct - 0.5) / 0.5) * (p05 - p10);
    } else if (pct <= 3.0) {
      estPrecision = p10 - ((pct - 1.0) / 2.0) * (p10 - p30);
    } else {
      estPrecision = Math.max(2.5, p30 - ((pct - 3.0) / 7.0) * 8.5);
    }
  }

  lbl.textContent = `Est. Precision: ${estPrecision.toFixed(1)}%`;
}

// Fetch KPI Stats
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();

    document.getElementById('kpi-total-events').textContent = (data.total_events_analyzed || 0).toLocaleString();
    document.getElementById('kpi-total-alerts').textContent = (data.total_alerts || 0).toLocaleString();
    document.getElementById('kpi-critical-sub').textContent = `${(data.critical_alerts_count || 0).toLocaleString()} High Risk (>=0.40)`;
    document.getElementById('kpi-cold-start').textContent = data.cold_start_alerts_count || 0;

    // Primary Threat
    const breakdown = data.attack_type_breakdown || {};
    let topThreat = 'None';
    let topCount = 0;
    for (const [type, count] of Object.entries(breakdown)) {
      if (type !== 'normal' && count > topCount) {
        topThreat = type;
        topCount = count;
      }
    }
    document.getElementById('kpi-top-threat').textContent = formatAttackType(topThreat);
    document.getElementById('kpi-threat-sub').textContent = `${topCount} incidents detected`;

  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}

// Load Alerts Queue Table with Pagination and Dynamic Counter Badge
async function loadAlerts() {
  const tbody = document.getElementById('alerts-tbody');
  if (!tbody) return;

  const search = document.getElementById('search-input')?.value || '';
  const type = document.getElementById('type-select')?.value || 'all';
  const minRisk = document.getElementById('risk-select')?.value || '0.0';
  const topPct = document.getElementById('alert-budget-slider')?.value || '1.0';

  let url = `/api/alerts?page=${currentPage}&limit=${pageSize}&top_pct=${topPct}&min_risk=${minRisk}&attack_type=${type}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    totalAlertsCount = data.total || 0;
    const totalPages = Math.ceil(totalAlertsCount / pageSize) || 1;

    // Update Counter Badge
    const countBadge = document.getElementById('alerts-count-badge');
    if (countBadge) {
      countBadge.textContent = `${totalAlertsCount.toLocaleString()} matching alerts found`;
    }

    // Update Pagination UI Bar
    const startIdx = totalAlertsCount > 0 ? (currentPage - 1) * pageSize + 1 : 0;
    const endIdx = Math.min(currentPage * pageSize, totalAlertsCount);

    const pagInfo = document.getElementById('pagination-info');
    if (pagInfo) {
      pagInfo.textContent = `Showing ${startIdx}-${endIdx} of ${totalAlertsCount.toLocaleString()} alerts`;
    }

    const pageDisplay = document.getElementById('page-num-display');
    if (pageDisplay) {
      pageDisplay.textContent = `Page ${currentPage} of ${totalPages}`;
    }

    const btnPrev = document.getElementById('btn-prev-page');
    const btnNext = document.getElementById('btn-next-page');

    if (btnPrev) btnPrev.disabled = currentPage <= 1;
    if (btnNext) btnNext.disabled = currentPage >= totalPages;

    if (!data.alerts || data.alerts.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 30px; color: var(--text-muted);">No alerts match the selected criteria.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.alerts.map(a => {
      const scoreObj = formatThreatScore(a.risk_score);
      const typeLabel = formatAttackType(a.predicted_type);
      const coldBadge = a.is_cold_start ? `<span class="state-badge badge-cold">COLD START</span>` : '';
      const driftBadge = a.is_drift_flagged ? `<span class="state-badge badge-drift">DRIFT</span>` : '';

      return `
        <tr onclick="openAlertDetail('${a.session_id}')">
          <td>${scoreObj.pillHtml}</td>
          <td><strong>${a.entity_id}</strong> ${coldBadge} ${driftBadge}</td>
          <td><span class="type-tag">${typeLabel}</span></td>
          <td style="color: var(--text-secondary); font-size: 0.82rem;">${formatTimestamp(a.timestamp)}</td>
          <td>${a.source_ip}<br><span style="font-size: 0.76rem; color: var(--text-muted);">${a.geo_location}</span></td>
          <td style="max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-secondary);">
            ${a.explanation_text || 'No explanation generated'}
          </td>
        </tr>
      `;
    }).join('');

  } catch (err) {
    console.error('Failed to load alerts:', err);
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--accent-red); padding: 20px;">Error loading alert queue.</td></tr>`;
  }
}

// Load Entities Table with Pagination & Filtering
async function loadEntities() {
  const tbody = document.getElementById('entities-tbody');
  if (!tbody) return;

  const search = document.getElementById('entity-search-input')?.value || '';
  const type = document.getElementById('entity-type-select')?.value || 'all';

  let url = `/api/entities?page=${entityCurrentPage}&limit=${entityPageSize}&entity_type=${type}`;
  if (search) url += `&search=${encodeURIComponent(search)}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    const entities = data.entities || [];
    totalEntitiesCount = data.total || 0;
    const totalPages = Math.ceil(totalEntitiesCount / entityPageSize) || 1;

    // Update Entity Counter Badge
    const countBadge = document.getElementById('entities-count-badge');
    if (countBadge) {
      countBadge.textContent = `${totalEntitiesCount.toLocaleString()} matching entity profiles found`;
    }

    // Update Entity Pagination UI Bar
    const startIdx = totalEntitiesCount > 0 ? (entityCurrentPage - 1) * entityPageSize + 1 : 0;
    const endIdx = Math.min(entityCurrentPage * entityPageSize, totalEntitiesCount);

    const pagInfo = document.getElementById('entity-pagination-info');
    if (pagInfo) {
      pagInfo.textContent = `Showing ${startIdx}-${endIdx} of ${totalEntitiesCount.toLocaleString()} entity profiles`;
    }

    const pageDisplay = document.getElementById('entity-page-num-display');
    if (pageDisplay) {
      pageDisplay.textContent = `Page ${entityCurrentPage} of ${totalPages}`;
    }

    const btnPrev = document.getElementById('btn-entity-prev-page');
    const btnNext = document.getElementById('btn-entity-next-page');

    if (btnPrev) btnPrev.disabled = entityCurrentPage <= 1;
    if (btnNext) btnNext.disabled = entityCurrentPage >= totalPages;

    if (entities.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; padding: 30px; color: var(--text-muted);">No entity profiles match the criteria.</td></tr>`;
      return;
    }

    tbody.innerHTML = entities.map(e => `
      <tr>
        <td><strong>${e.entity_id}</strong></td>
        <td><span class="type-tag" style="background: rgba(255,255,255,0.05); color: #fff; border-color: rgba(255,255,255,0.1);">${formatAttackType(e.entity_type)}</span></td>
        <td>${e.primary_auth_method || 'password'}</td>
        <td>Mean: ${e.hour_mean}h (std: ${e.hour_std})</td>
        <td>${(e.home_geo_set || []).join(', ') || 'N/A'}</td>
        <td><strong style="color: ${e.alert_count > 0 ? 'var(--accent-red)' : 'var(--accent-emerald)'};">${e.alert_count} alerts</strong></td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('Failed to load entities:', err);
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--accent-red); padding: 20px;">Error loading entity profiles.</td></tr>`;
  }
}

// Open Detail Modal with Clean Feature Attribution Progress Bars
async function openAlertDetail(sessionId) {
  const modal = document.getElementById('detail-modal');
  const modalTitle = document.getElementById('modal-title');
  const modalBody = document.getElementById('modal-body');

  modal.style.display = 'flex';
  modalTitle.textContent = `Alert Detail (Session ID: ${sessionId})`;
  modalBody.innerHTML = `<p style="padding: 20px; color: var(--text-muted);">Loading details...</p>`;

  try {
    const res = await fetch(`/api/alerts/${sessionId}`);
    const alert = await res.json();

    const topFeatures = alert.top_features || [];

    const cleanRawJson = { ...alert };
    if (cleanRawJson.top_features) delete cleanRawJson.top_features;

    const scoreObj = formatThreatScore(alert.risk_score);

    modalBody.innerHTML = `
      <div style="margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
        <div>
          <span class="type-tag" style="font-size: 0.9rem; padding: 6px 14px;">${formatAttackType(alert.predicted_type)}</span>
          <span style="margin-left: 12px;">Threat Score: ${scoreObj.fullHtml}</span>
        </div>
        <div style="font-size: 0.82rem; color: var(--text-secondary);">
          Entity: <strong style="color: #fff;">${alert.entity_id}</strong> (${alert.entity_type})
        </div>
      </div>

      <div class="glass-card" style="background: rgba(59, 130, 246, 0.08); border-color: rgba(59, 130, 246, 0.3); margin-bottom: 24px; padding: 18px 20px;">
        <h4 style="color: #93c5fd; margin-bottom: 6px; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.05em;">Analyst Explanation & Evidence</h4>
        <p style="font-size: 0.95rem; line-height: 1.5; color: #fff;">${alert.explanation_text || 'No explanation generated'}</p>
      </div>

      <div class="detail-grid">
        <div>
          <h4 style="margin-bottom: 12px; font-size: 0.88rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">Top Feature Contributions</h4>
          <div style="display: flex; flex-direction: column; gap: 10px;">
            ${topFeatures.map(f => {
              const featName = formatFeatureName(f[0]);
              const val = parseFloat(f[1]);
              const fillPct = Math.min(100, Math.max(10, val > 1 ? val * 25 : val * 100));
              return `
                <div class="feature-item">
                  <div class="feature-header">
                    <span style="color: #e2e8f0; font-weight: 500;">${featName}</span>
                    <strong style="color: var(--accent-cyan);">${val}</strong>
                  </div>
                  <div class="feature-bar-bg">
                    <div class="feature-bar-fill" style="width: ${fillPct}%;"></div>
                  </div>
                </div>
              `;
            }).join('') || '<div style="color: var(--text-muted);">No top feature attributions</div>'}
          </div>
        </div>

        <div>
          <h4 style="margin-bottom: 12px; font-size: 0.88rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">Raw Event JSON</h4>
          <pre class="json-box">${JSON.stringify(cleanRawJson, null, 2)}</pre>
        </div>
      </div>
    `;
  } catch (err) {
    modalBody.innerHTML = `<p style="color: var(--accent-red); padding: 20px;">Failed to load alert details.</p>`;
  }
}

// Helpers
function formatAttackType(type) {
  if (!type) return 'Unknown';
  return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatFeatureName(feat) {
  const mapping = {
    'distinct_entities_from_ip_5min': 'Cross-entity IP Concurrency',
    'failed_auth_count_5min': 'Failed Auth Velocity (5min)',
    'baseline_score': 'Baseline Anomaly Score',
    'resource_novelty': 'Resource Access Novelty',
    'geo_novelty': 'Geographic Novelty',
    'hour_zscore': 'Off-hour Z-Score',
    'duration_zscore': 'Session Duration Z-Score',
    'resource_breadth_new_1hr': 'Lateral Resource Access (1hr)',
    'geo_velocity_kmh': 'Travel Speed (km/h)',
    'offhours_access_trend_7d': 'Multi-day Off-hours Rate',
    'privilege_footprint_growth_30d': 'Privilege Footprint Growth'
  };
  return mapping[feat] || feat.replace(/_/g, ' ');
}

function formatTimestamp(ts) {
  if (!ts) return '';
  return ts.replace('T', ' ');
}

function formatThreatScore(score) {
  const val = parseFloat(score || 0);
  const score100 = (val * 100).toFixed(1);
  let level = 'LOW';
  let badgeClass = 'risk-low';

  if (val >= 0.80) {
    level = 'CRITICAL';
    badgeClass = 'risk-critical';
  } else if (val >= 0.40) {
    level = 'HIGH';
    badgeClass = 'risk-high';
  } else if (val >= 0.25) {
    level = 'MEDIUM';
    badgeClass = 'risk-med';
  }

  return {
    score100: score100,
    level: level,
    badgeClass: badgeClass,
    pillHtml: `<span class="risk-pill ${badgeClass}">${score100} / 100</span>`,
    fullHtml: `<span class="risk-pill ${badgeClass}">${score100} / 100 (${level})</span>`
  };
}

// Attack Simulation Modal Controller
function openSimulationModal() {
  const modal = document.getElementById('detail-modal');
  const modalTitle = document.getElementById('modal-title');
  if (!modal || !modalTitle) return;

  modal.style.display = 'flex';
  modalTitle.innerHTML = `⚡ Live Cyber Attack Simulation Engine`;
  renderSimulationControls(null);
}

function renderSimulationControls(simResult) {
  const modalBody = document.getElementById('modal-body');
  if (!modalBody) return;

  let resultHtml = '';
  if (simResult && simResult.simulated_alert) {
    const a = simResult.simulated_alert;
    let topFeats = [];
    if (typeof a.top_features === 'string') {
      try { topFeats = JSON.parse(a.top_features); } catch(e){}
    } else if (Array.isArray(a.top_features)) {
      topFeats = a.top_features;
    }

    const simScoreObj = formatThreatScore(a.risk_score);

    resultHtml = `
      <div class="glass-card" style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.4); margin-bottom: 20px; padding: 16px 20px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <h4 style="color: #6ee7b7; margin: 0; font-size: 0.95rem; font-weight: 700;">⚡ ${simResult.message || 'Attack Injected Successfully'}</h4>
          <span class="results-badge" style="background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.4);">LIVE INJECTION ACTIVE</span>
        </div>
      </div>

      <div class="glass-card" style="background: rgba(15, 23, 42, 0.7); margin-bottom: 20px;">
        <h4 style="color: var(--accent-cyan); margin-bottom: 12px; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.05em;">Injected Telemetry & Incident Details</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; font-size: 0.88rem;">
          <div><span style="color: var(--text-muted);">Attack Category:</span> <br><strong style="color: #93c5fd;">${formatAttackType(a.predicted_type)}</strong></div>
          <div><span style="color: var(--text-muted);">Session ID:</span> <br><code style="color: #e2e8f0;">${a.session_id}</code></div>
          <div><span style="color: var(--text-muted);">Threat Score (0-100):</span> <br>${simScoreObj.fullHtml}</div>
          <div><span style="color: var(--text-muted);">Target Entity:</span> <br><strong style="color: #fff;">${a.entity_id}</strong> (${a.entity_type})</div>
          <div><span style="color: var(--text-muted);">Source IP & Geo:</span> <br><span style="color: #fff;">${a.source_ip}</span> (${a.geo_location})</div>
          <div><span style="color: var(--text-muted);">Target Resource:</span> <br><span style="color: #fcd34d;">${a.resource_accessed}</span></div>
        </div>
      </div>

      <div class="glass-card" style="background: rgba(239, 68, 68, 0.08); border-color: rgba(239, 68, 68, 0.3); margin-bottom: 20px; padding: 16px 20px;">
        <h4 style="color: #fca5a5; margin-bottom: 6px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">AI Anomaly Explanation</h4>
        <p style="font-size: 0.92rem; line-height: 1.5; color: #fff; margin: 0;">${a.explanation_text}</p>
      </div>

      <div style="margin-bottom: 24px;">
        <h4 style="margin-bottom: 12px; font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">Key Feature Attributions</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${topFeats.map(f => {
            const featName = formatFeatureName(f[0]);
            const val = parseFloat(f[1]);
            const fillPct = Math.min(100, Math.max(10, val > 1 ? val * 25 : val * 100));
            return `
              <div class="feature-item">
                <div class="feature-header">
                  <span style="color: #e2e8f0; font-weight: 500;">${featName}</span>
                  <strong style="color: var(--accent-cyan);">${val}</strong>
                </div>
                <div class="feature-bar-bg">
                  <div class="feature-bar-fill" style="width: ${fillPct}%;"></div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;
  }

  modalBody.innerHTML = `
    <div style="margin-bottom: 20px; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: var(--radius-md); padding: 16px;">
      <label style="display: block; font-size: 0.88rem; font-weight: 600; color: #93c5fd; margin-bottom: 8px;">Select Cyber Attack Pattern to Inject:</label>
      <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
        <select id="sim-attack-select" class="select-styled" style="flex: 1; min-width: 260px; font-size: 0.9rem; padding: 10px 14px; background: #0f172a; border-color: rgba(255,255,255,0.2);">
          <option value="brute_force">🔴 Brute Force (High Frequency Auth Failure)</option>
          <option value="impossible_travel">⚡ Impossible Travel (Geographic Discrepancy)</option>
          <option value="credential_stuffing">🔑 Credential Stuffing (Multi-Account Velocity)</option>
          <option value="lateral_movement">🌐 Lateral Movement (Internal Privilege Escalation)</option>
        </select>
        <button id="btn-execute-sim" class="btn-page" style="background: linear-gradient(135deg, #ef4444, #dc2626); border: none; color: #fff; padding: 10px 22px; font-weight: 700; border-radius: 8px; cursor: pointer;">
          ⚡ Inject Attack
        </button>
      </div>
    </div>

    ${resultHtml}

    <div style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 20px; border-top: 1px solid var(--card-border); padding-top: 16px;">
      ${simResult ? `<button onclick="closeModalAndShowQueue()" class="btn-page" style="background: rgba(6, 182, 212, 0.25); border-color: var(--accent-cyan); color: #fff; font-weight: 600; padding: 8px 20px;">View in Alert Queue &raquo;</button>` : ''}
      <button onclick="closeModal()" class="btn-page" style="padding: 8px 20px;">Close</button>
    </div>
  `;

  document.getElementById('btn-execute-sim')?.addEventListener('click', async () => {
    const selectedType = document.getElementById('sim-attack-select')?.value || 'brute_force';
    const btn = document.getElementById('btn-execute-sim');
    if (btn) { btn.disabled = true; btn.textContent = 'Injecting...'; }

    try {
      const res = await fetch(`/api/simulate-attack?type=${selectedType}`, { method: 'POST' });
      const data = await res.json();
      currentPage = 1;
      loadStats();
      loadAlerts();
      renderSimulationControls(data);
    } catch (err) {
      alert('Attack simulation failed.');
    }
  });
}

function closeModal() {
  const modal = document.getElementById('detail-modal');
  if (modal) modal.style.display = 'none';
}

function closeModalAndShowQueue() {
  closeModal();
  switchView('alerts-view');
  window.scrollTo({ top: 400, behavior: 'smooth' });
}
