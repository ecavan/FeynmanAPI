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
  "e+ e- -> tau+ tau-|QED|0":
    "Electron-positron annihilation to tau pairs via s-channel photon. Same structure as e⁺e⁻ → μ⁺μ⁻ but with tau mass effects visible near threshold (√s ≈ 2m_τ ≈ 3.55 GeV).",
  "mu+ mu- -> mu+ mu-|QED|0":
    "Muon Bhabha scattering: the heavy-lepton analog of e⁺e⁻ → e⁺e⁻, with both s-channel annihilation and t-channel exchange. Relevant for muon collider physics.",
  "tau+ tau- -> e+ e-|QED|0":
    "Tau pair annihilation to electrons via s-channel photon. Tests lepton universality — the matrix element is structurally identical to e⁺e⁻ → μ⁺μ⁻.",
  "tau+ tau- -> mu+ mu-|QED|0":
    "Tau pair annihilation to muons via s-channel photon. Another lepton-universality test.",
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
  "e+ e- -> e+ e-|QED|1":
    "One-loop corrections to Bhabha scattering: vacuum polarization insertions in both s-channel and t-channel photon propagators, plus vertex corrections and box diagrams.",
  "e- e- -> e- e-|QED|1":
    "One-loop corrections to Møller scattering: vacuum polarization in t-channel and u-channel photon propagators, vertex corrections, and box diagrams.",
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
  "d d~ -> g g|QCD|0":
    "Down quark-antiquark annihilation to gluons. Identical Feynman structure to uū → gg — only the quark mass differs (negligible at high energy).",
  "d d~ -> s s~|QCD|0":
    "Down quark pair annihilation to strange quarks via s-channel gluon exchange.",
  "g g -> t t~|QCD|0":
    "Gluon fusion to top quarks — the dominant top pair production mechanism at the LHC. Top mass effects are significant since m_t ≈ 173 GeV.",
  // QCD flavour variants
  "g g -> c c~|QCD|0":
    "Gluon fusion to charm quarks. Important for open-charm production at the LHC and charm-tagged jet studies.",
  "b b~ -> g g|QCD|0":
    "Bottom quark-antiquark annihilation to gluons. The crossing-symmetric partner of gg → bb̄.",
  "u s~ -> u s~|QCD|0":
    "Different-flavour quark-antiquark scattering via t-channel gluon exchange only (no s-channel annihilation for different flavours in pure QCD).",
  // QCD+QED mixed
  "u u~ -> gamma gamma|QCDQED|0":
    "Quark-antiquark annihilation to two photons. A QCD+QED mixed process at O(α²) — no gluon exchange, pure QED-like topology but with coloured initial states.",
  "u u~ -> gamma g|QCDQED|0":
    "Associated photon + jet production. An O(αα_s) process important for prompt-photon measurements at hadron colliders.",
  "u g -> u gamma|QCDQED|0":
    "QCD Compton with photon emission: a quark scatters off a gluon and radiates a photon. The dominant source of isolated prompt photons at the LHC.",
  "gamma g -> u u~|QCDQED|0":
    "Photoproduction of quark pairs: a photon fuses with a gluon to produce a quark-antiquark pair. Important in γp collisions at HERA.",
  "d d~ -> gamma gamma|QCDQED|0":
    "Down quark-antiquark annihilation to two photons. Same topology as uū → γγ but with charge -1/3 quarks (coupling scaled by Q_d²).",
  "d d~ -> gamma g|QCDQED|0":
    "Associated photon + jet from down-quark annihilation. Probes the photon coupling to down-type quarks.",
  "d g -> d gamma|QCDQED|0":
    "Down-quark QCD Compton with prompt photon emission.",
  "gamma g -> d d~|QCDQED|0":
    "Photoproduction of down-quark pairs via photon-gluon fusion.",
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
  "u g -> u g|QCD|1":
    "One-loop corrections to quark-gluon Compton scattering: virtual gluon exchanges, vertex corrections, self-energy insertions, and box diagrams with full SU(3) color structure.",
  "g g -> u u~|QCD|1":
    "One-loop virtual corrections to gluon fusion into quark pairs.",
  // QED 1-loop extras
  "e+ e- -> gamma gamma|QED|1":
    "One-loop corrections to electron-positron annihilation into photon pairs: includes vertex corrections and box diagrams.",
  "gamma gamma -> e+ e-|QED|1":
    "One-loop corrections to photon-photon pair production: the crossed version of e⁺e⁻ → γγ with virtual fermion loops.",
  // EW 1-loop
  "Z -> e+ e-|EW|1":
    "One-loop corrections to Z decay including electroweak vertex corrections, self-energy insertions, and oblique corrections.",

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
  "t -> b W+|EW|0":
    "Top quark decay to bottom + W: the dominant (~100%) top decay mode. The top decays so fast (τ ~ 5×10⁻²⁵ s) that it never hadronizes — making it the only bare quark observable in experiment.",
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
  "e- nu_e~ -> W- Z|EW|0":
    "WZ associated production from electron-antineutrino scattering.",
  "tau+ tau- -> Z H|EW|0":
    "Higgsstrahlung from tau pairs.",
  "e+ e- -> W+ W- Z|EW|0":
    "Triple electroweak boson production: a 2→3 process testing quartic gauge couplings.",
  // Drell-Yan & Neutrino
  "u d~ -> e+ nu_e|EW|0":
    "Charged-current Drell-Yan: a u-quark and d̄-antiquark annihilate via an s-channel W⁺ boson to produce a positron and neutrino. The primary W production mechanism at hadron colliders and the process used to discover the W at the SPS in 1983.",
  "e- nu_mu -> mu- nu_e|EW|0":
    "Inverse muon decay: neutrino-electron scattering via t-channel W exchange. A purely charged-current process used to measure the Fermi constant and test lepton universality.",
  // EW 1-loop
  "e+ e- -> mu+ mu-|EW|1":
    "One-loop electroweak corrections to e⁺e⁻ → μ⁺μ⁻ including W/Z loops, self-energy insertions, and vertex corrections with the full γ+Z propagator structure.",

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

  // ── Additional QED 1-loop ──
  "gamma gamma -> gamma gamma|QED|1":
    "Light-by-light scattering: a 1-loop QED process with no tree-level contribution. Predicted by Euler & Heisenberg in 1936; first observed at the LHC in 2017 by ATLAS and CMS in ultra-peripheral Pb-Pb collisions.",
  "e+ e- -> t t~|QED|1":
    "QED 1-loop box contribution to top-pair production at high-energy lepton colliders (CEPC/ILC physics). Captures the QED-only piece; full SM also includes Z exchange and weak loops.",
  "q q~ -> g g|QCD|1":
    "Full 1-loop virtual amplitude for qq̄ → gg, including IR-divergent box, vertex, and self-energy diagrams. Source: Catani, NPB 478 (1996).",

  // ── EW: qq̄ → ZH per quark flavor ──
  "u u~ -> Z H|EW|0":
    "uū → ZH (Higgsstrahlung) — leading channel for pp → ZH at the LHC. Hagiwara-Zeppenfeld formula with full V-A Z couplings.",
  "d d~ -> Z H|EW|0":
    "dd̄ → ZH — second-largest channel for pp → ZH (about 50% of uū contribution).",
  "c c~ -> Z H|EW|0":
    "cc̄ → ZH — small contribution, suppressed by charm PDF.",
  "s s~ -> Z H|EW|0":
    "ss̄ → ZH — suppressed by strange PDF.",
  "b b~ -> Z H|EW|0":
    "bb̄ → ZH — small but distinctive contribution; bb̄ initial states are also relevant for HVH searches.",

  // ── EW: qq̄ → ZZ per quark flavor ──
  "u u~ -> Z Z|EW|0":
    "uū → ZZ via t-channel quark exchange. Computed via a numerical 8-γ trace evaluator (closed-form analogue of e⁺e⁻ → ZZ).",
  "d d~ -> Z Z|EW|0":
    "dd̄ → ZZ — uses the same 8-γ trace evaluator with down-type Z couplings.",
  "c c~ -> Z Z|EW|0":
    "cc̄ → ZZ — small charm-PDF contribution to pp → ZZ.",
  "s s~ -> Z Z|EW|0":
    "ss̄ → ZZ — small strange-PDF contribution.",
  "b b~ -> Z Z|EW|0":
    "bb̄ → ZZ — relevant for HVZ searches with b-tagging.",

  // ── EW: qq̄ → W+W- per quark flavor ──
  "u u~ -> W+ W-|EW|0":
    "uū → W⁺W⁻ via t-channel d-quark exchange. The largest pp → WW partonic channel at the LHC. Engine uses Hagiwara-Peccei-Zeppenfeld t-channel only — see README for the s-channel γ+Z+TGC limitation.",
  "d d~ -> W+ W-|EW|0":
    "dd̄ → W⁺W⁻ via t-channel u-quark exchange.",
  "c c~ -> W+ W-|EW|0":
    "cc̄ → W⁺W⁻ via t-channel s-quark.",
  "s s~ -> W+ W-|EW|0":
    "ss̄ → W⁺W⁻ via t-channel c-quark.",
  "b b~ -> W+ W-|EW|0":
    "bb̄ → W⁺W⁻ via t-channel t-quark exchange — significant due to large m_t coupling.",

  // ── EW: radiative DY ──
  "u d~ -> W+ gamma|EW|0":
    "Radiative charged-current DY: ud̄ → W⁺γ with the WWγ triple-gauge coupling. Has a celebrated radiation zero at cosθ* = 1/3.",

  // ── EW: W- charge variants ──
  "W- -> e- nu_e~|EW|0":
    "W⁻ leptonic decay to electron + ν̄ₑ. Same partial width as W⁺ → e⁺νₑ by CP.",
  "W- -> mu- nu_mu~|EW|0":
    "W⁻ → μ⁻ν̄μ — universal with W⁻ → e⁻ν̄ₑ in the massless-lepton limit.",
  "W- -> tau- nu_tau~|EW|0":
    "W⁻ → τ⁻ν̄τ — small mass correction from m_τ.",
  "W- -> d u~|EW|0":
    "W⁻ hadronic decay to du̅ (CKM-diagonal). Multiply by N_c = 3 for the colour-summed rate; total hadronic BR(W) ≈ 67%.",

  // ── EW: loop-induced Higgs decays ──
  "H -> gamma gamma|EW|1":
    "Loop-induced H → γγ: vanishing at tree level, generated by W and top loops. Discovery channel for the Higgs at the LHC (2012). PDG BR ≈ 0.227%.",
  "H -> g g|QCD|1":
    "Loop-induced H → gg via top-quark loop (other quark loops are Yukawa-suppressed). Drives gluon-fusion Higgs production by reciprocity.",
  "H -> Z gamma|EW|1":
    "Loop-induced H → Zγ via W and top loops. Rare decay (BR ≈ 0.15%) recently reported by ATLAS+CMS.",
  "H -> b b~|EW|1":
    "QCD NLO correction to H → bb̄: K = 1 + 17 α_s/(3π) from the curated 1-loop QCD vertex correction to the Yukawa coupling.",

  // ── EW: loop-induced production ──
  "g g -> H|EW|1":
    "Gluon fusion Higgs production via the full top-quark triangle loop (NOT the heavy-top effective theory). Drives ~90% of pp → H at the LHC.",

  // ── BSM Z' interference channels ──
  "e+ e- -> e+ e-|BSM|0":
    "Bhabha scattering with both γ and Z' s+t-channel exchange. The Z' modifies the cross-section by interference and a possible resonance peak — the discovery channel for new neutral gauge bosons at e⁺e⁻ colliders.",
  "e- e- -> e- e-|BSM|0":
    "Møller scattering with t+u-channel γ and Z' exchange. Used by SLAC E158 / future MOLLER to probe new physics through parity-violating asymmetries that include Z' contributions.",
  "mu+ mu- -> mu+ mu-|BSM|0":
    "μ Bhabha at a future muon collider with γ + Z' exchange. Sensitive to muon-philic Z' models that could explain the (g-2)_μ anomaly.",
  "e+ e- -> mu+ mu-|BSM|0":
    "e⁺e⁻ → μ⁺μ⁻ with both γ and Z' s-channel mediators. Resonant Z' enhancement appears as a dilepton mass peak — the prototypical Z' search at LHC and future leptonic colliders.",
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
const tabNav         = document.getElementById("tab-nav");
const tabPanelDiagrams      = document.getElementById("tab-diagrams");
const tabPanelDistributions = document.getElementById("tab-distributions");
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
const filterUniqueTopo = document.getElementById("filter-unique-topo");
const amplitudeSection = document.getElementById("amplitude-section");
const amplitudeFormula = document.getElementById("amplitude-formula");
const amplitudeNotes   = document.getElementById("amplitude-notes");
const integralSection  = document.getElementById("integral-section");
const integralFormula  = document.getElementById("integral-formula");
const integralNotes    = document.getElementById("integral-notes");
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
  // Sync the theory dropdown with the sidebar tab
  if (theorySelect.value !== filter) {
    // Add option if it doesn't exist (e.g. QCDQED before API loads)
    if (![...theorySelect.options].some(o => o.value === filter)) {
      const opt = document.createElement("option");
      opt.value = filter;
      opt.textContent = filter;
      theorySelect.appendChild(opt);
    }
    theorySelect.value = filter;
  }
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
            <th>PDG ID</th>
            <th>Mass (MeV)</th>
            <th>Width (MeV)</th>
            <th>Charge</th>
          </tr>
        </thead>
        <tbody>
    `;

    particles.forEach(p => {
      const mass = p.mass_mev !== null ? p.mass_mev.toFixed(3) : (p.mass || "0");
      const width = p.width_mev !== null && p.width_mev !== undefined
        ? (p.width_mev > 0 ? p.width_mev.toFixed(3) : "stable")
        : "\u2014";
      const charge = p.charge !== null ? p.charge : "0";
      const pdgId = p.pdg_id !== null && p.pdg_id !== undefined ? p.pdg_id : "\u2014";
      const displayName = p.latex_name
        ? `<span title="${p.pdg_name || p.name}">${p.name}</span>`
        : p.name;
      html += `
        <tr>
          <td class="name">${displayName}</td>
          <td class="pdg-id">${pdgId}</td>
          <td class="mass">${mass}</td>
          <td class="width">${width}</td>
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
          no_tadpoles:       filterNotadpole.checked,
          one_pi:            filterOnePi.checked,
          connected:         true,
          unique_topologies: filterUniqueTopo.checked,
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
    fetchAmplitude(process, theorySelect.value, loopOrder);

  } catch (err) {
    showError(String(err));
  } finally {
    setLoading(false);
  }
}

// ── Amplitude ─────────────────────────────────────────────────────────────────
async function fetchAmplitude(process, theory, loops = 0) {
  try {
    const res = await fetch(
      `${API_BASE}/amplitude?process=${encodeURIComponent(process)}&theory=${encodeURIComponent(theory)}&loops=${loops}`
    );
    const data = await _readJson(res);
    if (!res.ok) {
      const message = data?.detail || "Tree-level |M|² and integral information are not available for this process.";
      showAmplitudeUnavailable(message);
      showIntegralUnavailable(message);
      return;
    }

    // ── |M|² section ─────────────────────────────────────────────────────────
    if (data.has_msq) {
      amplitudeFormula.classList.remove("formula-unavailable");
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
      const details = [];
      if (data.description) details.push(data.description);
      if (data.notes) details.push(data.notes);
      amplitudeNotes.textContent = details.join("  ·  ");
      amplitudeSection.classList.remove("hidden");
    } else {
      showAmplitudeUnavailable(
        data.notes ||
        data.availability_message ||
        "Spin-averaged |M|² is not available yet for this process."
      );
    }

    // ── Integral section ──────────────────────────────────────────────────────
    if (data.has_integral) {
      _renderIntegral(data.integral_latex || null);
    } else {
      showIntegralUnavailable(
        data.availability_message ||
        "An integral representation is not available yet for this process."
      );
    }
  } catch (_) {
    showAmplitudeUnavailable("Tree-level |M|² is not available right now.");
    showIntegralUnavailable("The integral representation is not available right now.");
  }
}

function _renderIntegral(latex_str) {
  if (!latex_str) {
    showIntegralUnavailable("An integral representation is not available yet for this process.");
    return;
  }
  integralFormula.classList.remove("formula-unavailable");
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
  integralNotes.textContent = "";
  integralSection.classList.remove("hidden");
}

// ── Loop Integral ─────────────────────────────────────────────────────────────
async function fetchLoopIntegral(process, theory, loops) {
  try {
    const res = await fetch(
      `${API_BASE}/amplitude/loop-integral?process=${encodeURIComponent(process)}&theory=${encodeURIComponent(theory)}&loops=${loops}`
    );
    const data = await _readJson(res);
    if (!res.ok) {
      showIntegralUnavailable(
        data?.detail || `A ${loops}-loop integral representation is not available for this process.`
      );
      return;
    }
    if (data.has_integral) {
      _renderIntegral(data.integral_latex || null);
    } else {
      showIntegralUnavailable(
        data.availability_message ||
        data.notes ||
        `A ${loops}-loop integral representation is not available for this process.`
      );
    }
  } catch (_) {
    showIntegralUnavailable("The loop integral representation is not available right now.");
  }
}

function showAmplitudeUnavailable(message) {
  amplitudeFormula.classList.add("formula-unavailable");
  amplitudeFormula.textContent = "|M|² is not available yet for this process.";
  amplitudeNotes.textContent = message || "";
  amplitudeSection.classList.remove("hidden");
}

function showIntegralUnavailable(message) {
  integralFormula.classList.add("formula-unavailable");
  integralFormula.textContent = "An integral representation is not available yet for this process.";
  integralNotes.textContent = message || "";
  integralSection.classList.remove("hidden");
}

async function _readJson(res) {
  try {
    return await res.json();
  } catch (_) {
    return null;
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
  tabNav.classList.remove("hidden");
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

// ── Differential histogram ────────────────────────────────────────────────────
const histComputeBtn = document.getElementById("hist-compute-btn");
const histStatus = document.getElementById("hist-status");
const histCanvasWrap = document.getElementById("hist-canvas-wrap");
const histSvg = document.getElementById("hist-svg");
const histSummary = document.getElementById("hist-summary");
const obsSelect = document.getElementById("obs-select");
const histSqrts = document.getElementById("hist-sqrts");
const histXmin = document.getElementById("hist-xmin");
const histXmax = document.getElementById("hist-xmax");
const histNbins = document.getElementById("hist-nbins");
const histOrder = document.getElementById("hist-order");
const histNevents = document.getElementById("hist-nevents");

// Auto-set default range when observable changes
obsSelect.addEventListener("change", () => {
  const obs = obsSelect.value;
  const defaults = {
    cos_theta:  { min: -1,  max: 1,    nbins: 20 },
    pT_lepton:  { min: 0,   max: 50,   nbins: 25 },
    pT_photon:  { min: 0,   max: 50,   nbins: 25 },
    eta_lepton: { min: -5,  max: 5,    nbins: 20 },
    y_lepton:   { min: -5,  max: 5,    nbins: 20 },
    M_inv:      { min: 0,   max: 200,  nbins: 25 },
    M_ll:       { min: 60,  max: 120,  nbins: 24 },
    DR_ll:      { min: 0,   max: 6.28, nbins: 20 },
  };
  const d = defaults[obs];
  if (d) {
    histXmin.value = d.min;
    histXmax.value = d.max;
    histNbins.value = d.nbins;
  }
});

histComputeBtn.addEventListener("click", computeHistogram);

async function computeHistogram() {
  const process = processInput.value.trim();
  if (!process) {
    histStatus.textContent = "Enter a process string above first.";
    return;
  }
  histStatus.textContent = "Computing… (MC sampling can take 10–60 seconds for 2→N)";
  histCanvasWrap.classList.add("hidden");
  histComputeBtn.disabled = true;

  const params = new URLSearchParams({
    process,
    theory: theorySelect.value,
    sqrt_s: histSqrts.value,
    observable: obsSelect.value,
    bin_min: histXmin.value,
    bin_max: histXmax.value,
    n_bins: histNbins.value,
    order: histOrder.value,
    n_events: histNevents.value,
  });

  try {
    const res = await fetch(`${API_BASE}/amplitude/differential-distribution?${params}`);
    const data = await res.json();
    if (!res.ok) {
      // Trust-blocked processes return 422 with a structured detail object
      const detail = data.detail;
      if (res.status === 422 && typeof detail === "object" && detail?.trust_level === "blocked") {
        histCanvasWrap.classList.remove("hidden");
        histSvg.innerHTML = "";
        const reasonHtml =
          `<div style="background:#fce4e4;border-left:4px solid #b30000;padding:0.85rem 1rem;border-radius:4px;color:#5a0000;">` +
          `<strong>✕ Process blocked</strong>` +
          (detail.block_reason ? `<div style="margin-top:0.4rem;">${escHtml(detail.block_reason)}</div>` : "") +
          (detail.workaround    ? `<div style="margin-top:0.4rem;font-size:0.92em;"><em>Workaround:</em> ${escHtml(detail.workaround)}</div>` : "") +
          `</div>`;
        histSummary.innerHTML = reasonHtml;
        histStatus.textContent = "";
        return;
      }
      histStatus.textContent = "Error: " + (typeof detail === "string" ? detail : JSON.stringify(detail) || res.statusText);
      histStatus.style.color = "var(--danger, #c33)";
      return;
    }
    histStatus.textContent = "";
    histStatus.style.color = "";
    renderHistogram(data);
  } catch (err) {
    histStatus.textContent = "Network error: " + err.message;
    histStatus.style.color = "var(--danger, #c33)";
  } finally {
    histComputeBtn.disabled = false;
  }
}

function renderHistogram(data) {
  const edges = data.bin_edges;
  const widths = data.bin_widths;
  const dy = data.dsigma_dX_pb;
  const err = data.dsigma_dX_uncertainty_pb || dy.map(() => 0);
  const obs = data.observable;
  const unit = data.unit || "";

  // SVG layout
  const svgW = 720, svgH = 360;
  const m = { l: 70, r: 20, t: 30, b: 60 };
  const w = svgW - m.l - m.r;
  const h = svgH - m.t - m.b;
  const xMin = edges[0], xMax = edges[edges.length - 1];
  const xRange = xMax - xMin || 1;

  // y range
  const yMaxRaw = Math.max(...dy.map((v, i) => Math.abs(v) + (err[i] || 0)));
  const yMax = yMaxRaw > 0 ? yMaxRaw * 1.10 : 1;

  const xPx = x => m.l + ((x - xMin) / xRange) * w;
  const yPx = y => m.t + h - (y / yMax) * h;

  let parts = [];
  // axes
  parts.push(`<line class="axis" x1="${m.l}" y1="${m.t + h}" x2="${m.l + w}" y2="${m.t + h}" />`);
  parts.push(`<line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${m.t + h}" />`);

  // x ticks
  const nTicks = 6;
  for (let i = 0; i <= nTicks; i++) {
    const xv = xMin + (i / nTicks) * xRange;
    const px = xPx(xv);
    parts.push(`<line class="axis" x1="${px}" y1="${m.t + h}" x2="${px}" y2="${m.t + h + 4}" />`);
    parts.push(`<text class="tick" x="${px}" y="${m.t + h + 16}" text-anchor="middle">${formatTick(xv)}</text>`);
  }
  // y ticks
  for (let i = 0; i <= 5; i++) {
    const yv = (i / 5) * yMax;
    const py = yPx(yv);
    parts.push(`<line class="axis" x1="${m.l - 4}" y1="${py}" x2="${m.l}" y2="${py}" />`);
    parts.push(`<text class="tick" x="${m.l - 6}" y="${py + 3}" text-anchor="end">${formatTick(yv)}</text>`);
  }

  // bars + error bars
  for (let i = 0; i < dy.length; i++) {
    const x0 = xPx(edges[i]);
    const x1 = xPx(edges[i + 1]);
    const barW = Math.max(x1 - x0 - 1, 1);
    const y0 = yPx(Math.max(dy[i], 0));
    const barH = Math.max(yPx(0) - y0, 0);
    parts.push(`<rect class="bar" x="${x0}" y="${y0}" width="${barW}" height="${barH}" />`);
    if (err[i] > 0) {
      const xc = (x0 + x1) / 2;
      const yu = yPx(Math.max(dy[i] + err[i], 0));
      const yl = yPx(Math.max(dy[i] - err[i], 0));
      parts.push(`<line class="errbar" x1="${xc}" y1="${yu}" x2="${xc}" y2="${yl}" />`);
      parts.push(`<line class="errbar" x1="${xc - 3}" y1="${yu}" x2="${xc + 3}" y2="${yu}" />`);
      parts.push(`<line class="errbar" x1="${xc - 3}" y1="${yl}" x2="${xc + 3}" y2="${yl}" />`);
    }
  }

  // axis labels
  const xLabel = unit ? `${obs} (${unit})` : obs;
  parts.push(`<text class="axis-label" x="${m.l + w / 2}" y="${svgH - 12}" text-anchor="middle">${xLabel}</text>`);
  parts.push(
    `<text class="axis-label" x="${15}" y="${m.t + h / 2}" text-anchor="middle" ` +
    `transform="rotate(-90 15 ${m.t + h / 2})">dσ/d(${obs}) [pb${unit ? "/" + unit : ""}]</text>`
  );
  parts.push(`<text class="axis-label" x="${m.l + w / 2}" y="${m.t - 10}" text-anchor="middle">` +
    `${escHtml(data.process)}, √s = ${data.sqrt_s_gev} GeV — ${data.method} (${data.order})</text>`);

  histSvg.innerHTML = parts.join("\n");

  // Summary text
  const sigmaTotal = data.sigma_total_pb;
  const summary = [];
  summary.push(`<strong>σ_total</strong> = ${sigmaTotal.toExponential(4)} pb`);
  summary.push(`<strong>order</strong>: ${data.order}`);
  summary.push(`<strong>method</strong>: ${data.method}`);
  if (data.k_factor) summary.push(`<strong>K</strong> = ${data.k_factor.toFixed(6)}`);
  if (data.pdf) summary.push(`<strong>PDF</strong>: <code>${escHtml(data.pdf)}</code>`);
  if (data.n_events) summary.push(`<strong>n_events</strong>: ${data.n_events}`);
  if (data.limitations) {
    summary.push(`<em style="color:#a60">limitations: ${escHtml(data.limitations)}</em>`);
  }
  // Trust badge + caveat banner (from /amplitude/cross-section trust system)
  const trustBadge = renderTrustBadge(data);
  histSummary.innerHTML = trustBadge + summary.join("  ·  ");

  histCanvasWrap.classList.remove("hidden");
}

// ── Trust badge / caveat rendering ───────────────────────────────────────────
function renderTrustBadge(data) {
  const level = data.trust_level;
  if (!level) return "";
  const colors = {
    validated:   { bg: "#e7f5ec", border: "#1e7d36", fg: "#0c4319", icon: "✓" },
    approximate: { bg: "#fff8e1", border: "#c98400", fg: "#7a4f00", icon: "≈" },
    rough:       { bg: "#ffe1cc", border: "#cc5500", fg: "#7a3300", icon: "!" },
    blocked:     { bg: "#fce4e4", border: "#b30000", fg: "#5a0000", icon: "✕" },
  };
  const c = colors[level] || colors.approximate;

  let banner = `<div style="background:${c.bg};border-left:4px solid ${c.border};padding:0.6rem 0.9rem;margin-bottom:0.7rem;border-radius:4px;font-size:0.9em;color:${c.fg}">`;
  banner += `<strong>${c.icon} Trust: ${level}</strong>`;
  if (data.trust_reference) {
    banner += ` &mdash; <span style="font-size:0.92em">${escHtml(data.trust_reference)}</span>`;
  }
  if (data.accuracy_caveat) {
    banner += `<div style="margin-top:0.35rem;font-size:0.92em">${escHtml(data.accuracy_caveat)}</div>`;
  }
  banner += `</div>`;
  return banner;
}

function formatTick(v) {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10)  return v.toFixed(1);
  if (Math.abs(v) >= 1)   return v.toFixed(2);
  if (Math.abs(v) >= 0.01) return v.toFixed(3);
  if (v === 0) return "0";
  return v.toExponential(1);
}

// ── Tab navigation ───────────────────────────────────────────────────────────
function activateTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    const isActive = btn.dataset.tab === tabId;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  tabPanelDiagrams.classList.toggle("hidden", tabId !== "diagrams");
  tabPanelDistributions.classList.toggle("hidden", tabId !== "distributions");
}

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});
