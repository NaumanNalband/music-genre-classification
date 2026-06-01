/**
 * script.js — Shared frontend logic for Music Genre Classification System
 * Handles: auth state, API calls, localStorage, nav highlights
 */

/* ─── Config ─────────────────────────────────────────────────── */
const API_BASE = window.location.origin + '/api';   // Flask serves both

/* ─── Auth Helpers ───────────────────────────────────────────── */

function getToken()  { return localStorage.getItem('mgc_token'); }
function getUser()   {
  try { return JSON.parse(localStorage.getItem('mgc_user') || 'null'); }
  catch { return null; }
}
function saveAuth(token, user) {
  localStorage.setItem('mgc_token', token);
  localStorage.setItem('mgc_user', JSON.stringify(user));
}
function clearAuth() {
  localStorage.removeItem('mgc_token');
  localStorage.removeItem('mgc_user');
}
function isLoggedIn() { return !!getToken(); }

/** Redirect to login if not authenticated. Call at top of protected pages. */
function requireAuth() {
  if (!isLoggedIn()) { window.location.href = 'index.html'; }
}
/** Redirect to dashboard if already logged in. Call on auth pages. */
function redirectIfLoggedIn() {
  if (isLoggedIn()) { window.location.href = 'dashboard.html'; }
}

/* ─── Fetch Wrapper ──────────────────────────────────────────── */

async function apiFetch(endpoint, options = {}) {
  const token = getToken();
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${endpoint}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));

  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/** FormData (multipart) upload — no Content-Type header override */
async function apiUpload(endpoint, formData) {
  const token = getToken();
  const headers = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST', headers, body: formData,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/* ─── UI Helpers ─────────────────────────────────────────────── */

function showAlert(container, message, type = 'error') {
  container.innerHTML = `
    <div class="alert alert-${type}">
      <span>${type === 'error' ? '⚠️' : type === 'success' ? '✅' : 'ℹ️'}</span>
      ${message}
    </div>`;
}

function setLoading(btn, loading, text = 'Processing…') {
  if (loading) {
    btn._origText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> ${text}`;
  } else {
    btn.disabled = false;
    btn.innerHTML = btn._origText || 'Submit';
  }
}

function formatDate(iso) {
  return new Date(iso).toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

/** Populate sidebar with logged-in user info */
function initSidebar() {
  const user = getUser();
  if (!user) return;

  document.querySelectorAll('[data-username]').forEach(el => el.textContent = user.username);
  document.querySelectorAll('[data-avatar]').forEach(el => el.textContent = user.username[0].toUpperCase());

  document.querySelectorAll('.logout-btn').forEach(btn => {
    btn.addEventListener('click', () => { clearAuth(); window.location.href = 'index.html'; });
  });

  // Hamburger toggle
  const ham = document.querySelector('.hamburger');
  const sidebar = document.querySelector('.sidebar');
  if (ham && sidebar) {
    ham.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', e => {
      if (!sidebar.contains(e.target) && e.target !== ham) sidebar.classList.remove('open');
    });
  }

  // Active nav link
  const current = window.location.pathname.split('/').pop() || 'dashboard.html';
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('href') === current) link.classList.add('active');
  });
}

/* ─── Genre Colours ──────────────────────────────────────────── */

const GENRE_COLORS = {
  blues:     '#4fc3f7',
  classical: '#ce93d8',
  country:   '#ffd54f',
  disco:     '#ff8a65',
  hiphop:    '#81c784',
  jazz:      '#4dd0e1',
  metal:     '#ef9a9a',
  pop:       '#f48fb1',
  reggae:    '#a5d6a7',
  rock:      '#ffcc02',
};

function genreColor(genre) {
  return GENRE_COLORS[genre?.toLowerCase()] || '#00e5ff';
}

/* ─── Waveform Decoration ────────────────────────────────────── */

function buildWaveform(container, bars = 20) {
  container.innerHTML = '';
  for (let i = 0; i < bars; i++) {
    const bar = document.createElement('div');
    bar.className = 'wave-bar';
    bar.style.height = Math.floor(Math.random() * 28 + 8) + 'px';
    container.appendChild(bar);
  }
}

/* ─── Run on every page ──────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', initSidebar);
