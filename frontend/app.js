/* FeynmanEngine frontend */

const API_BASE = "/api";

// ── State ────────────────────────────────────────────────────────────────────
let lastResult = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const processInput   = document.getElementById("process-input");
const theorySelect   = document.getElementById("theory-select");
const loopsSelect    = document.getElementById("loops-select");
const generateBtn    = document.getElementById("generate-btn");
const exportBtn      = document.getElementById("export-btn");
const validationMsg  = document.getElementById("validation-msg");
const resultsSection = document.getElementById("results-section");
const resultsTitle   = document.getElementById("results-title");
const resultsMeta    = document.getElementById("results-meta");
const diagramGrid    = document.getElementById("diagram-grid");
const loading        = document.getElementById("loading");
const errorBox       = document.getElementById("error-box");
const modalOverlay   = document.getElementById("modal-overlay");
const modalClose     = document.getElementById("modal-close");
const modalDiagram   = document.getElementById("modal-diagram-view");
const modalMetadata  = document.getElementById("modal-metadata");
const modalTikzCode  = document.getElementById("modal-tikz-code");
const copyTikzBtn    = document.getElementById("copy-tikz-btn");
const modalDlGroup   = document.getElementById("modal-dl-group");
const themeToggle      = document.getElementById("theme-toggle");
const filterNotadpole  = document.getElementById("filter-notadpole");
const filterOnePi      = document.getElementById("filter-onepi");
const amplitudeSection = document.getElementById("amplitude-section");
const amplitudeFormula = document.getElementById("amplitude-formula");
const amplitudeNotes   = document.getElementById("amplitude-notes");

// ── Theme ─────────────────────────────────────────────────────────────────────
function applyTheme(dark) {
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  localStorage.setItem("theme", dark ? "dark" : "light");
}

themeToggle.addEventListener("click", () => {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  applyTheme(!isDark);
});

// Load saved theme
applyTheme(localStorage.getItem("theme") === "dark");

// ── Load theories from API ────────────────────────────────────────────────────
async function loadTheories() {
  try {
    const res = await fetch(`${API_BASE}/theories`);
    if (!res.ok) return;
    const theories = await res.json();
    theorySelect.innerHTML = "";
    theories.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      theorySelect.appendChild(opt);
    });
  } catch (_) { /* server may not be up yet */ }
}

loadTheories();

// ── Theory filter tabs ────────────────────────────────────────────────────────
document.querySelectorAll(".theory-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".theory-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const filter = tab.dataset.filter;
    document.querySelectorAll(".example-group").forEach(group => {
      const theory = group.dataset.theory;
      group.style.display = (filter === "all" || theory === filter) ? "" : "none";
    });
  });
});

// ── Example process buttons ───────────────────────────────────────────────────
document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    processInput.value = btn.dataset.process;
    theorySelect.value = btn.dataset.theory;
    loopsSelect.value  = btn.dataset.loops;
    validationMsg.textContent = "";
    document.querySelectorAll(".example-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    generate();
  });
});

// ── Live validation ───────────────────────────────────────────────────────────
let validateTimer = null;

processInput.addEventListener("input", () => {
  clearTimeout(validateTimer);
  validationMsg.textContent = "";
  validateTimer = setTimeout(validateProcess, 600);
});

async function validateProcess() {
  const process = processInput.value.trim();
  const theory  = theorySelect.value;
  if (!process) return;

  try {
    const res = await fetch(
      `${API_BASE}/describe?process=${encodeURIComponent(process)}&theory=${encodeURIComponent(theory)}`
    );
    if (res.ok) {
      validationMsg.textContent = "";
      validationMsg.style.color = "var(--accent)";
    } else {
      const body = await res.json();
      validationMsg.textContent = body.detail || "Invalid process.";
      validationMsg.style.color = "var(--danger)";
    }
  } catch (_) {
    validationMsg.textContent = "";
  }
}

// ── Generate ──────────────────────────────────────────────────────────────────
generateBtn.addEventListener("click", generate);
processInput.addEventListener("keydown", e => { if (e.key === "Enter") generate(); });

async function generate() {
  const process = processInput.value.trim();
  if (!process) {
    validationMsg.textContent = "Enter a process string first.";
    return;
  }

  setLoading(true);
  hideError();
  resultsSection.classList.add("hidden");
  amplitudeSection.classList.add("hidden");
  exportBtn.disabled = true;
  lastResult = null;

  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        process,
        theory: theorySelect.value,
        loops: parseInt(loopsSelect.value, 10),
        output_format: "svg",
        filters: {
          no_tadpoles: filterNotadpole.checked,
          one_pi:      filterOnePi.checked,
          connected:   true,
        },
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || JSON.stringify(data));
      return;
    }

    lastResult = data;
    renderResults(data);
    exportBtn.disabled = false;
    fetchAmplitude(process, theorySelect.value);

  } catch (err) {
    showError(String(err));
  } finally {
    setLoading(false);
  }
}

// ── Amplitude ─────────────────────────────────────────────────────────────────
async function fetchAmplitude(process, theory) {
  try {
    const res = await fetch(
      `${API_BASE}/amplitude?process=${encodeURIComponent(process)}&theory=${encodeURIComponent(theory)}`
    );
    if (!res.ok) { amplitudeSection.classList.add("hidden"); return; }
    const data = await res.json();
    if (!data.msq_sympy) { amplitudeSection.classList.add("hidden"); return; }
    amplitudeFormula.textContent = `|M|² = ${data.msq_sympy}`;
    const desc  = data.description || "";
    const notes = data.notes || "";
    amplitudeNotes.textContent = desc + (notes ? "  ·  " + notes : "");
    amplitudeSection.classList.remove("hidden");
  } catch (_) {
    amplitudeSection.classList.add("hidden");
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const { diagrams, summary, metadata } = data;

  resultsTitle.textContent = `${summary.total_diagrams} diagram${summary.total_diagrams !== 1 ? "s" : ""}`;
  const tops = Object.entries(summary.topology_counts || {})
    .map(([k, v]) => `${v}× ${k}`)
    .join(", ");
  resultsMeta.textContent =
    `${metadata.theory} · ${metadata.process} · ${metadata.loops === 0 ? "tree-level" : `${metadata.loops}-loop`}` +
    (tops ? ` · ${tops}` : "") +
    (metadata.elapsed_seconds ? ` · ${metadata.elapsed_seconds}s` : "");

  diagramGrid.innerHTML = "";

  if (diagrams.length === 0) {
    diagramGrid.innerHTML = `<p style="color:var(--text-muted);font-size:.85rem">No diagrams generated.</p>`;
  } else {
    diagrams.forEach(d => {
      const card = buildDiagramCard(d);
      diagramGrid.appendChild(card);
    });
  }

  resultsSection.classList.remove("hidden");
}

function buildDiagramCard(d) {
  const card = document.createElement("div");
  card.className = "diagram-card";
  card.dataset.diagramId = d.id;

  const imgBox = document.createElement("div");
  imgBox.className = "diagram-img";

  if (d.image_b64) {
    const img = document.createElement("img");
    img.src = `data:image/svg+xml;base64,${d.image_b64}`;
    img.alt = `Diagram ${d.id}`;
    imgBox.appendChild(img);
  } else if (d.tikz_code) {
    // Show a TikZ snippet placeholder
    const pre = document.createElement("div");
    pre.className = "diagram-placeholder";
    pre.textContent = "TikZ\n(no render)";
    imgBox.appendChild(pre);
  } else {
    imgBox.innerHTML = `<div class="diagram-placeholder">No image</div>`;
  }

  const meta = document.createElement("div");
  meta.className = "diagram-card-meta";

  const numSpan = document.createElement("div");
  numSpan.className = "diagram-num";
  numSpan.textContent = `Diagram ${d.id}`;

  const right = document.createElement("div");
  right.style.cssText = "display:flex;align-items:center;gap:6px";

  if (d.topology) {
    const tag = document.createElement("span");
    tag.className = "topology-tag";
    tag.textContent = d.topology;
    right.appendChild(tag);
  }

  if (d.image_b64) {
    const dlGroup = document.createElement("div");
    dlGroup.className = "download-group";
    dlGroup.innerHTML = `
      <button class="btn-dl" data-dl="svg" title="Download SVG">SVG</button>
      <button class="btn-dl" data-dl="png" title="Download PNG">PNG</button>
      <button class="btn-dl" data-dl="pdf" title="Download PDF">PDF</button>
    `;
    // Stop click from bubbling to card (which opens modal)
    dlGroup.addEventListener("click", e => {
      e.stopPropagation();
      const action = e.target.dataset.dl;
      if (action === "svg") downloadSVG(d);
      else if (action === "png") downloadPNG(d);
      else if (action === "pdf") downloadPDF(d.id, e.target);
    });
    right.appendChild(dlGroup);
  }

  meta.appendChild(numSpan);
  meta.appendChild(right);

  card.appendChild(imgBox);
  card.appendChild(meta);

  card.addEventListener("click", () => openModal(d));

  return card;
}

// ── Download helpers ──────────────────────────────────────────────────────────
function _triggerDownload(url, filename) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function downloadSVG(d) {
  const svgBytes = atob(d.image_b64);
  const arr = new Uint8Array(svgBytes.length);
  for (let i = 0; i < svgBytes.length; i++) arr[i] = svgBytes.charCodeAt(i);
  const blob = new Blob([arr], { type: "image/svg+xml" });
  _triggerDownload(URL.createObjectURL(blob), `diagram_${d.id}.svg`);
}

async function downloadPNG(d) {
  const svgDataUrl = `data:image/svg+xml;base64,${d.image_b64}`;
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = rej;
    img.src = svgDataUrl;
  });
  // 4× upscale for crisp PNG at print resolution
  const scale = 4;
  const canvas = document.createElement("canvas");
  canvas.width  = (img.naturalWidth  || img.width  || 300) * scale;
  canvas.height = (img.naturalHeight || img.height || 300) * scale;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.scale(scale, scale);
  ctx.drawImage(img, 0, 0);
  _triggerDownload(canvas.toDataURL("image/png"), `diagram_${d.id}.png`);
}

async function downloadPDF(diagramId, btn) {
  const prev = btn.textContent;
  btn.textContent = "…";
  btn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/diagram/${diagramId}/pdf`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`PDF failed: ${err.detail || res.statusText}`);
      return;
    }
    const blob = await res.blob();
    _triggerDownload(URL.createObjectURL(blob), `diagram_${diagramId}.pdf`);
  } catch (e) {
    alert(`PDF error: ${e}`);
  } finally {
    btn.textContent = prev;
    btn.disabled = false;
  }
}

// ── Modal ─────────────────────────────────────────────────────────────────────
function openModal(d) {
  modalDiagram.innerHTML = "";

  if (d.image_b64) {
    const img = document.createElement("img");
    img.src = `data:image/svg+xml;base64,${d.image_b64}`;
    img.alt = `Diagram ${d.id}`;
    modalDiagram.appendChild(img);
  } else {
    modalDiagram.innerHTML = `<div class="diagram-placeholder" style="padding:2rem">No rendered image</div>`;
  }

  modalMetadata.innerHTML = `
    <dt>Diagram</dt>     <dd>${d.id}</dd>
    <dt>Process</dt>     <dd>${escHtml(d.process)}</dd>
    <dt>Theory</dt>      <dd>${d.theory}</dd>
    <dt>Loop order</dt>  <dd>${d.loop_order}</dd>
    <dt>Topology</dt>    <dd>${d.topology || "—"}</dd>
    <dt>Sym. factor</dt> <dd>${d.symmetry_factor ?? "—"}</dd>
    <dt>Vertices</dt>    <dd>${d.n_vertices}</dd>
    <dt>Edges</dt>       <dd>${d.n_edges}</dd>
  `;

  modalTikzCode.textContent = d.tikz_code || "(no TikZ code)";

  // Populate modal download buttons
  modalDlGroup.innerHTML = "";
  if (d.image_b64) {
    modalDlGroup.innerHTML = `
      <button class="btn-dl" data-dl="svg">SVG</button>
      <button class="btn-dl" data-dl="png">PNG</button>
      <button class="btn-dl" data-dl="pdf">PDF</button>
    `;
    modalDlGroup.onclick = e => {
      const action = e.target.dataset.dl;
      if (action === "svg") downloadSVG(d);
      else if (action === "png") downloadPNG(d);
      else if (action === "pdf") downloadPDF(d.id, e.target);
    };
  }

  modalOverlay.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

modalClose.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", e => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

function closeModal() {
  modalOverlay.classList.add("hidden");
  document.body.style.overflow = "";
}

// ── Copy TikZ ─────────────────────────────────────────────────────────────────
copyTikzBtn.addEventListener("click", () => {
  const text = modalTikzCode.textContent;
  navigator.clipboard.writeText(text).then(() => {
    copyTikzBtn.textContent = "Copied!";
    setTimeout(() => { copyTikzBtn.textContent = "Copy"; }, 1500);
  });
});

// ── Export all TikZ ───────────────────────────────────────────────────────────
exportBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const parts = lastResult.diagrams
    .map(d => `% === Diagram ${d.id} (${d.topology || "unknown"}) ===\n${d.tikz_code || ""}`)
    .join("\n\n");
  const blob = new Blob([parts], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "feynman_diagrams.tex";
  a.click();
  URL.revokeObjectURL(url);
});

// ── UI helpers ────────────────────────────────────────────────────────────────
function setLoading(on) {
  loading.classList.toggle("hidden", !on);
  generateBtn.disabled = on;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function escHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
