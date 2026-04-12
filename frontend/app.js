/* FeynmanEngine frontend */

const API_BASE = "/api";

// ── State ────────────────────────────────────────────────────────────────────
let lastResult = null;

// ── Process descriptions (educational context for example processes) ──────────
const PROCESS_DESCRIPTIONS = {
  // ── QED ──
  "e+ e- -> mu+ mu-|QED|0":
    "Electron-positron annihilation into muon pairs via a single virtual photon (s-channel). This is the simplest non-trivial QED scattering process and a cornerstone of collider physics — it was used at LEP to precisely measure the Z boson mass.",
  "e+ e- -> e+ e-|QED|0":
    "Bhabha scattering: elastic electron-positron scattering proceeding through both s-channel annihilation and t-channel photon exchange. The interference between these two diagrams produces a distinctive forward peak used for luminosity monitoring at e⁺e⁻ colliders.",
  "e- gamma -> e- gamma|QED|0":
    "Compton scattering: a photon scatters off an electron via s-channel and u-channel electron exchange. The Klein-Nishina formula describing its cross section was one of the earliest successes of QED.",
  "e- e- -> e- e-|QED|0":
    "Møller scattering: electron-electron scattering via t-channel and u-channel photon exchange. The identical fermion final state requires careful antisymmetrization, producing a characteristic interference pattern.",
  "e+ e- -> gamma gamma|QED|0":
    "Pair annihilation into two photons. The reverse of pair production, this process proceeds through t-channel and u-channel electron exchange and is forbidden for a single photon by energy-momentum conservation.",
  "gamma gamma -> e+ e-|QED|0":
    "Breit-Wheeler pair production: two photons collide to create an electron-positron pair. First predicted in 1934, direct observation of this process in vacuum remains an active experimental goal.",
  "mu+ mu- -> gamma gamma|QED|0":
    "Muon pair annihilation into two photons — the heavy-lepton analog of e⁺e⁻ → γγ. Identical structure to the electron case but with muon mass effects in the propagators.",
  "e- mu- -> e- mu-|QED|0":
    "Electron-muon elastic scattering via single t-channel photon exchange. Because the two fermion flavors are distinguishable, there is no u-channel contribution — making this the cleanest probe of the QED vertex.",
  "e+ mu- -> e+ mu-|QED|0":
    "Positron-muon scattering via t-channel photon exchange. Like e⁻μ⁻ scattering but with crossed initial-state fermion lines.",
  "mu+ mu- -> e+ e-|QED|0":
    "Muon pair annihilation to electron pairs via s-channel photon. The crossing-symmetric partner of e⁺e⁻ → μ⁺μ⁻.",
  // QED Radiative
  "e+ e- -> mu+ mu- gamma|QED|0":
    "Initial- or final-state radiation in muon pair production: a real photon is emitted alongside the μ⁺μ⁻ pair. This 2→3 process is essential for understanding radiative corrections.",
  "e+ e- -> e+ e- gamma|QED|0":
    "Radiative Bhabha scattering with an extra photon emission. Important for luminosity calibration at colliders.",
  "e- gamma -> e- gamma gamma|QED|0":
    "Double Compton scattering: an electron emits an additional photon during Compton scattering, producing a 2→3 final state.",
  "mu+ mu- -> e+ e- gamma|QED|0":
    "Radiative muon annihilation into an electron pair plus a photon.",
  // QED 1-loop
  "e+ e- -> mu+ mu-|QED|1":
    "One-loop corrections to e⁺e⁻ → μ⁺μ⁻ including vertex corrections, vacuum polarization (fermion loops in the photon propagator), and box diagrams. These are the first quantum corrections to the tree-level cross section.",
  "e- gamma -> e- gamma|QED|1":
    "One-loop corrections to Compton scattering, involving vertex corrections, self-energy insertions, and box diagrams.",

  // ── QCD ──
  "u u~ -> g g|QCD|0":
    "Quark-antiquark annihilation into two gluons via t/u-channel quark exchange and s-channel gluon exchange. A key QCD process at hadron colliders with nontrivial SU(3) color structure.",
  "g g -> g g|QCD|0":
    "Gluon-gluon scattering: proceeds through s, t, u-channel diagrams plus a 4-gluon contact interaction unique to non-Abelian gauge theories. Dominates the jet cross section at the LHC.",
  "g g -> u u~|QCD|0":
    "Gluon fusion into a quark-antiquark pair — the dominant mechanism for heavy quark (top, bottom) production at the LHC.",
  "g g -> b b~|QCD|0":
    "Gluon fusion to bottom quarks. Important for b-jet production and Higgs searches in the H → bb̄ channel.",
  "u u~ -> u u~|QCD|0":
    "Quark-antiquark elastic scattering via gluon exchange (both s and t channels).",
  "u d~ -> u d~|QCD|0":
    "Quark-antiquark scattering of different flavors — only t-channel gluon exchange contributes (no s-channel annihilation for different flavors in pure QCD).",
  "u u~ -> d d~|QCD|0":
    "Quark flavor change via s-channel gluon exchange. Pure QCD process that probes the gluon propagator.",
  "u g -> u g|QCD|0":
    "QCD Compton scattering: a quark scatters off a gluon. Analog of QED Compton but with color structure.",
  "u u -> u u|QCD|0":
    "Identical-quark scattering via t-channel and u-channel gluon exchange, requiring proper antisymmetrization.",
  "u u~ -> s s~|QCD|0":
    "Quark-antiquark annihilation into strange quarks via s-channel gluon.",
  // QCD multi-leg
  "u u~ -> g g g|QCD|0":
    "Three-gluon production from quark annihilation — a 2→3 QCD process with rich color structure.",
  "g g -> u u~ g|QCD|0":
    "Gluon fusion with an additional gluon radiated. Important for NLO jet calculations.",
  "g g -> g g g|QCD|0":
    "Five-gluon amplitude: a tour de force of QCD perturbation theory involving triple and quartic gluon vertices.",
  "u g -> u g g|QCD|0":
    "Quark-gluon scattering with additional gluon emission.",
  // QCD 1-loop
  "u u~ -> g g|QCD|1":
    "One-loop corrections to qq̄ → gg including virtual gluon loops and ghost contributions from gauge fixing.",
  "g g -> g g|QCD|1":
    "One-loop gluon-gluon scattering: involves gluon, ghost, and quark loops. A classic test of non-Abelian gauge theory.",

  // ── EW ──
  "Z -> e+ e-|EW|0":
    "Z boson decay to an electron-positron pair. Measured with exquisite precision at LEP, the partial width constrains the weak mixing angle sin²θ_W.",
  "Z -> mu+ mu-|EW|0":
    "Z → μ⁺μ⁻: the muon channel of Z decay, used to verify lepton universality.",
  "Z -> tau+ tau-|EW|0":
    "Z → τ⁺τ⁻: tau lepton production from Z decay.",
  "Z -> b b~|EW|0":
    "Z → bb̄: the dominant hadronic Z decay mode. Its measurement at LEP provided an early constraint on the top quark mass via loop corrections.",
  "Z -> nu_e nu_e~|EW|0":
    "Z → νν̄: invisible Z decay. Counting the invisible width at LEP established that there are exactly three light neutrino generations.",
  "W+ -> e+ nu_e|EW|0":
    "Leptonic W decay: the W boson decays to a positron and electron neutrino. Clean experimental signature used for W mass measurements.",
  "W+ -> mu+ nu_mu|EW|0":
    "W⁺ → μ⁺νμ: muon channel of W decay.",
  "W+ -> tau+ nu_tau|EW|0":
    "W⁺ → τ⁺ντ: tau channel of W decay, important for lepton universality tests.",
  // Higgs Decays
  "H -> b b~|EW|0":
    "Higgs decay to bottom quarks — the dominant decay mode (~58% branching ratio). First observed at the LHC in 2018 by ATLAS and CMS.",
  "H -> tau+ tau-|EW|0":
    "H → τ⁺τ⁻: Yukawa coupling of the Higgs to tau leptons, confirming the Higgs mechanism generates lepton masses.",
  "H -> c c~|EW|0":
    "H → cc̄: Higgs decay to charm quarks. Very challenging to observe due to the small Yukawa coupling and large QCD backgrounds.",
  "H -> mu+ mu-|EW|0":
    "H → μ⁺μ⁻: extremely rare Higgs decay (~0.02% BR) that directly probes the second-generation Yukawa coupling. Evidence reported by CMS in 2020.",
  "H -> W+ W-|EW|0":
    "Higgs decay to W boson pairs — one of the discovery channels at the LHC (2012). One W is typically off-shell for m_H = 125 GeV.",
  "H -> Z Z|EW|0":
    "H → ZZ*: the 'golden channel' for Higgs discovery due to the clean four-lepton final state.",
  // Weak Decays
  "mu- -> e- nu_e~ nu_mu|EW|0":
    "Muon decay: the prototypical charged-current weak decay, described by the Fermi theory at low energies. Its lifetime determines the Fermi constant G_F.",
  "tau- -> mu- nu_mu~ nu_tau|EW|0":
    "Tau decay to muon: tests lepton universality between the second and third generation.",
  "tau- -> e- nu_e~ nu_tau|EW|0":
    "Tau decay to electron: another lepton universality test.",
  // EW Scattering
  "e+ e- -> mu+ mu-|EW|0":
    "e⁺e⁻ → μ⁺μ⁻ in the full electroweak theory: both photon and Z boson exchange contribute. Near the Z pole, the Z diagram dominates, producing the famous Z resonance peak.",
  "e+ e- -> W+ W-|EW|0":
    "W pair production: involves neutrino t-channel exchange, s-channel γ/Z exchange, and the triple gauge coupling WWγ/WWZ. Precise cancellations between diagrams are required by gauge invariance.",
  "e+ e- -> Z H|EW|0":
    "Higgsstrahlung: the primary Higgs production mechanism at e⁺e⁻ colliders. A Z boson radiates a Higgs boson. This process was searched for at LEP and will be the workhorse of future Higgs factories.",
  "e+ e- -> Z Z|EW|0":
    "Z pair production at e⁺e⁻ colliders.",
  "e+ e- -> t t~|EW|0":
    "Top pair production at e⁺e⁻ colliders via s-channel γ/Z exchange. Enables precision measurements of the top quark mass and electroweak couplings.",
  "e+ e- -> H H|EW|0":
    "Double Higgs production: sensitive to the Higgs self-coupling λ, a key parameter for understanding the shape of the Higgs potential and electroweak symmetry breaking.",
  "e- nu_e -> W- Z|EW|0":
    "WZ associated production from electron-neutrino scattering.",
  "tau+ tau- -> Z H|EW|0":
    "Higgsstrahlung from tau pairs.",
  "e+ e- -> W+ W- Z|EW|0":
    "Triple electroweak boson production: a 2→3 process testing quartic gauge couplings.",

  // ── BSM ──
  "Zp -> e+ e-|BSM|0":
    "Z' decay to electrons: a signature dilepton resonance searched for at the LHC to discover new neutral gauge bosons predicted by many BSM theories (E₆, SO(10), etc.).",
  "Zp -> mu+ mu-|BSM|0":
    "Z' decay to muons: the muon channel of Z' dilepton searches.",
  "Zp -> chi chi~|BSM|0":
    "Z' decay to dark matter particles (χχ̄): if kinematically allowed, this invisible decay mode could explain the relic abundance of dark matter.",
  "e+ e- -> chi chi~|BSM|0":
    "Dark matter pair production via Z' mediator: a collider signature for simplified dark matter models. Would appear as missing energy at e⁺e⁻ colliders.",
  "mu+ mu- -> chi chi~|BSM|0":
    "DM production from muon annihilation via Z' exchange.",
  "e+ e- -> Zp Zp|BSM|0":
    "Z' pair production: possible if the Z'-fermion coupling is large enough. Probes the dark sector at high-energy colliders.",
  "chi chi~ -> e+ e-|BSM|0":
    "Dark matter annihilation to electrons via Z' mediator — relevant for indirect detection experiments searching for DM annihilation products in cosmic rays.",
  "chi chi~ -> mu+ mu-|BSM|0":
    "DM annihilation to muons.",
  "chi chi~ -> chi chi~|BSM|0":
    "Dark matter self-scattering via Z' exchange: determines the DM self-interaction cross section, which affects halo density profiles and is constrained by observations of galaxy cluster mergers (Bullet Cluster).",
};

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
const integralSection  = document.getElementById("integral-section");
const integralFormula  = document.getElementById("integral-formula");
const processDescBox   = document.getElementById("process-description");
const particleTable    = document.getElementById("particle-properties-table");

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
    // Set initial filter to QED (first theory)
    updateTheoryFilter("QED");
  } catch (_) { /* server may not be up yet */ }
}

loadTheories();

// Sync theory select with sidebar filter
theorySelect.addEventListener("change", () => {
  updateTheoryFilter(theorySelect.value);
});

// ── Theory filter tabs ────────────────────────────────────────────────────────
function updateTheoryFilter(filter) {
  document.querySelectorAll(".theory-tab").forEach(t => {
    t.classList.toggle("active", t.dataset.filter === filter);
  });
  document.querySelectorAll(".example-group").forEach(group => {
    const theory = group.dataset.theory;
    group.style.display = (theory === filter) ? "" : "none";
  });
  loadParticles(filter);
}

document.querySelectorAll(".theory-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    updateTheoryFilter(tab.dataset.filter);
  });
});
// ── Particle Reference ────────────────────────────────────────────────────────
async function loadParticles(theory) {
  if (!particleTable) return;
  particleTable.innerHTML = `<div class="loading-small">Loading ${theory} particles...</div>`;
  
  try {
    const res = await fetch(`${API_BASE}/theories/${theory}/particles`);
    if (!res.ok) {
      particleTable.innerHTML = `<div class="loading-small">Failed to load particles.</div>`;
      return;
    }
    const particles = await res.json();
    
    if (particles.length === 0) {
      particleTable.innerHTML = `<div class="loading-small">No particle data available.</div>`;
      return;
    }

    let html = `
      <table class="particle-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Mass (MeV)</th>
            <th>Charge</th>
          </tr>
        </thead>
        <tbody>
    `;

    particles.forEach(p => {
      const mass = p.mass_mev !== null ? p.mass_mev.toFixed(3) : (p.mass || "0");
      const charge = p.charge !== null ? p.charge : "0";
      html += `
        <tr>
          <td class="name">${p.name}</td>
          <td class="mass">${mass}</td>
          <td class="charge">${charge}</td>
        </tr>
      `;
    });

    html += `</tbody></table>`;
    particleTable.innerHTML = html;
  } catch (err) {
    particleTable.innerHTML = `<div class="loading-small">Error: ${err.message}</div>`;
  }
}

// ── Example process buttons ───────────────────────────────────────────────────
document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    processInput.value = btn.dataset.process;
    theorySelect.value = btn.dataset.theory;
    loopsSelect.value  = btn.dataset.loops;
    validationMsg.textContent = "";
    document.querySelectorAll(".example-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    // Show description for this example if available.
    const descKey = `${btn.dataset.process}|${btn.dataset.theory}|${btn.dataset.loops}`;
    const desc = PROCESS_DESCRIPTIONS[descKey];
    if (desc) {
      processDescBox.textContent = desc;
      processDescBox.classList.remove("hidden");
    } else {
      processDescBox.classList.add("hidden");
    }

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
  integralSection.classList.add("hidden");
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

    const loopOrder = parseInt(loopsSelect.value, 10);
    if (loopOrder === 0) {
      fetchAmplitude(process, theorySelect.value);
    } else {
      // For loop diagrams, fetch the loop integral representation.
      amplitudeSection.classList.add("hidden");
      fetchLoopIntegral(process, theorySelect.value, loopOrder);
    }

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
    if (!res.ok) { amplitudeSection.classList.add("hidden"); integralSection.classList.add("hidden"); return; }
    const data = await res.json();

    // ── |M|² section ─────────────────────────────────────────────────────────
    // Only show when msq_sympy is non-empty (i.e. full amplitude was computed).
    if (data.msq_sympy) {
      if (data.msq_latex && typeof katex !== "undefined") {
        try {
          katex.render(`|\\mathcal{M}|^2 = ${data.msq_latex}`, amplitudeFormula, {
            displayMode: true,
            throwOnError: false,
            trust: true,
          });
        } catch (_) {
          amplitudeFormula.textContent = `|M|² = ${data.msq_sympy}`;
        }
      } else {
        amplitudeFormula.textContent = `|M|² = ${data.msq_sympy}`;
      }
      const desc  = data.description || "";
      const notes = data.notes || "";
      amplitudeNotes.textContent = desc + (notes ? "  ·  " + notes : "");
      amplitudeSection.classList.remove("hidden");
    } else {
      amplitudeSection.classList.add("hidden");
    }

    // ── Integral section ──────────────────────────────────────────────────────
    // Show whenever integral_latex is present, even if |M|² wasn't computed.
    _renderIntegral(data.integral_latex);
  } catch (_) {
    amplitudeSection.classList.add("hidden");
    integralSection.classList.add("hidden");
  }
}

function _renderIntegral(latex_str) {
  if (!latex_str) { integralSection.classList.add("hidden"); return; }
  if (typeof katex !== "undefined") {
    try {
      katex.render(latex_str, integralFormula, {
        displayMode: true,
        throwOnError: false,
        trust: true,
      });
    } catch (_) {
      integralFormula.textContent = latex_str;
    }
  } else {
    integralFormula.textContent = latex_str;
  }
  integralSection.classList.remove("hidden");
}

// ── Loop Integral ─────────────────────────────────────────────────────────────
async function fetchLoopIntegral(process, theory, loops) {
  try {
    const res = await fetch(
      `${API_BASE}/amplitude/loop-integral?process=${encodeURIComponent(process)}&theory=${encodeURIComponent(theory)}&loops=${loops}`
    );
    if (!res.ok) { integralSection.classList.add("hidden"); return; }
    const data = await res.json();
    _renderIntegral(data.integral_latex || null);
  } catch (_) {
    integralSection.classList.add("hidden");
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
