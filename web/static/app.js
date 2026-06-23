const STORAGE_KEY = "tris_api_key";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function getApiKey() {
  return localStorage.getItem(STORAGE_KEY) || "";
}

function setApiKey(value) {
  if (value) localStorage.setItem(STORAGE_KEY, value);
  else localStorage.removeItem(STORAGE_KEY);
}

async function apiFetch(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;

  const res = await fetch(path, { ...options, headers });
  if (res.status === 401) {
    showToast("API key required or invalid", true);
    $("#api-key-dialog").showModal();
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

function showToast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, 4000);
}

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(1)}%`;
}

function fmtOdds(triple) {
  if (!triple || triple.length < 3) return "—";
  return triple.map((x) => Number(x).toFixed(2)).join(" / ");
}

function leagueName(code) {
  return { E0: "EPL", SP1: "La Liga", D1: "Bundesliga", I1: "Serie A", F1: "Ligue 1" }[code] || code;
}

function renderStats(container, items) {
  container.innerHTML = items
    .map(
      ({ label, value, cls = "" }) =>
        `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value ${cls}">${value}</div></div>`
    )
    .join("");
}

function renderCacheBars(caches, completed) {
  const max = Math.max(completed, 1);
  const rows = [
    ["Understat", caches.understat],
    ["StatsBomb", caches.statsbomb],
    ["FBref", caches.fbref],
    ["Chaos", caches.chaos],
  ];
  $("#cache-bars").innerHTML = rows
    .map(([name, n]) => {
      const pct = Math.min(100, (n / max) * 100);
      return `<div class="cache-row"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><span>${n.toLocaleString()}</span></div>`;
    })
    .join("");
}

async function loadHealth() {
  const pill = $("#health-pill");
  try {
    const data = await apiFetch("/health");
    pill.textContent = data.status;
    pill.className = "pill pill-ok";
  } catch {
    pill.textContent = "offline";
    pill.className = "pill pill-err";
  }
}

async function loadAuthConfig() {
  const cfg = await fetch("/auth/config").then((r) => r.json());
  const btn = $("#api-key-btn");
  if (cfg.auth_required) {
    btn.hidden = false;
    if (!getApiKey()) $("#api-key-dialog").showModal();
  }
}

async function loadStatus() {
  const grid = $("#stat-grid");
  grid.classList.add("loading");
  try {
    const data = await apiFetch("/status");
    const fr = data.fixture_readiness;
    $("#fixture-guidance").textContent = fr.guidance;

    renderStats(grid, [
      { label: "Matches", value: data.matches_total.toLocaleString() },
      { label: "Completed", value: data.matches_completed.toLocaleString() },
      { label: "Upcoming Big 5", value: String(fr.upcoming_big5), cls: fr.upcoming_big5 ? "gold" : "" },
      {
        label: "Live ready",
        value: fr.ready_for_live_predict ? "Yes" : "No",
        cls: fr.ready_for_live_predict ? "positive" : "",
      },
    ]);
    renderCacheBars(data.caches, data.matches_completed);
  } catch (e) {
    showToast(e.message, true);
  } finally {
    grid.classList.remove("loading");
  }
}

async function loadFixtures() {
  const limit = $("#fixtures-limit").value;
  const tbody = $("#fixtures-table tbody");
  const empty = $("#fixtures-empty");
  tbody.innerHTML = "";
  empty.hidden = true;

  try {
    const rows = await apiFetch(`/fixtures/upcoming?limit=${limit}`);
    if (!rows.length) {
      empty.hidden = false;
      return;
    }
    tbody.innerHTML = rows
      .map(
        (r) =>
          `<tr><td class="league">${leagueName(r.div)}</td><td>${r.date}</td><td>${r.home_team}</td><td>${r.away_team}</td><td>${fmtOdds([r.b365_h, r.b365_d, r.b365_a])}</td></tr>`
      )
      .join("");
  } catch (e) {
    if (e.message !== "Unauthorized") showToast(e.message, true);
  }
}

async function runPredictions(ev) {
  ev.preventDefault();
  const form = ev.target;
  const btn = form.querySelector('button[type="submit"]');
  const list = $("#predictions-list");
  const msg = $("#predict-message");
  list.innerHTML = "";
  msg.textContent = "Running model…";
  btn.disabled = true;

  const params = new URLSearchParams({
    confidence: form.confidence.value,
    train_limit: form.train_limit.value,
    predict_limit: form.predict_limit.value,
    dry_run: form.dry_run.checked,
  });

  try {
    const data = await apiFetch(`/predictions?${params}`);
    msg.textContent = data.message || `${data.predictions.length} edge-qualified pick(s).`;
    if (!data.predictions.length) return;

    list.innerHTML = data.predictions
      .map((p) => {
        const edge = p.edge != null ? `${(p.edge * 100).toFixed(1)}% edge` : "";
        const probs = p.probs
          ? `H ${(p.probs.H * 100).toFixed(0)}% · D ${(p.probs.D * 100).toFixed(0)}% · A ${(p.probs.A * 100).toFixed(0)}%`
          : "";
        return `<article class="pick-card"><h4>${p.home} vs ${p.away} <span class="league">[${leagueName(p.div || "")}]</span></h4><p class="pick-outcome">${p.outcome} · ${p.confidence.toFixed(1)}% confidence</p><div class="pick-meta"><span>${p.date}</span><span><strong>${edge}</strong></span><span>~${p.expected_goals?.toFixed(1) ?? "?"} xG</span><span>${probs}</span><span>Bookie: ${p.bookie_pick || "—"}</span><span>Close: ${fmtOdds(p.b365_close)}</span></div></article>`;
      })
      .join("");
  } catch (e) {
    msg.textContent = "";
    if (e.message !== "Unauthorized") showToast(e.message, true);
  } finally {
    btn.disabled = false;
  }
}

async function runBacktest(ev) {
  ev.preventDefault();
  const form = ev.target;
  const btn = form.querySelector('button[type="submit"]');
  const container = $("#backtest-metrics");
  btn.disabled = true;
  container.classList.add("loading");

  try {
    const data = await apiFetch(`/backtest?limit=${form.limit.value}`);
    const m = data.metrics;
    const roiCls = m.selective_roi_pct >= 0 ? "positive" : "negative";
    renderStats(container, [
      { label: "Holdout accuracy", value: fmtPct(m.holdout_accuracy_pct) },
      { label: "Bookie baseline", value: fmtPct(m.bookie_accuracy_pct) },
      { label: "Selective accuracy", value: fmtPct(m.selective_accuracy_pct) },
      { label: "Selective ROI", value: fmtPct(m.selective_roi_pct), cls: roiCls },
      { label: "Selective picks", value: String(m.selective_picks) },
      { label: "All picks", value: String(m.all_picks) },
      { label: "Train / test", value: `${m.train_matches ?? "?"} / ${m.test_matches ?? "?"}` },
    ]);
    showToast("Backtest complete");
  } catch (e) {
    if (e.message !== "Unauthorized") showToast(e.message, true);
  } finally {
    btn.disabled = false;
    container.classList.remove("loading");
  }
}

function initTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      $$(".panel").forEach((p) => {
        p.classList.remove("active");
        p.hidden = true;
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      const panel = $(`#panel-${tab.dataset.panel}`);
      panel.classList.add("active");
      panel.hidden = false;
    });
  });
}

function initApiKeyDialog() {
  $("#api-key-btn")?.addEventListener("click", () => {
    $("#api-key-input").value = getApiKey();
    $("#api-key-dialog").showModal();
  });
  $("#api-key-clear")?.addEventListener("click", () => {
    setApiKey("");
    $("#api-key-input").value = "";
    showToast("API key cleared");
  });
  $("#api-key-form")?.addEventListener("submit", (ev) => {
    ev.preventDefault();
    setApiKey($("#api-key-input").value.trim());
    $("#api-key-dialog").close();
    showToast("API key saved");
    loadStatus();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initApiKeyDialog();
  $("#refresh-status")?.addEventListener("click", loadStatus);
  $("#load-fixtures")?.addEventListener("click", loadFixtures);
  $("#predict-form")?.addEventListener("submit", runPredictions);
  $("#backtest-form")?.addEventListener("submit", runBacktest);

  loadAuthConfig();
  loadHealth();
  loadStatus();
});