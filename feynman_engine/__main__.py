"""CLI entry point for FeynmanEngine."""
import sys


def _run_setup(
    force: bool,
    include_looptools: bool,
    include_openloops: bool = False,
) -> int:
    from feynman_engine.form import build_form
    from feynman_engine.lhapdf import build_lhapdf, install_pdf_set, DEFAULT_PDF_SET
    from feynman_engine.looptools import build_looptools
    from feynman_engine.qgraf import build_qgraf

    steps = [
        ("QGRAF", build_qgraf),
        ("FORM", build_form),
    ]
    if include_looptools:
        steps.append(("LoopTools", build_looptools))

    def _build_lhapdf_and_default_set(force=False):
        prefix = build_lhapdf(force=force)
        try:
            install_pdf_set(DEFAULT_PDF_SET, prefix=prefix)
        except Exception as exc:
            print(f"[setup] LHAPDF built; default set {DEFAULT_PDF_SET} install failed: {exc}")
        return prefix

    steps.append(("LHAPDF", _build_lhapdf_and_default_set))
    if include_openloops:
        from feynman_engine.openloops import (
            build_openloops, install_process_library,
            DEFAULT_PROCESS_LIBRARY, DEFAULT_EW_NLO_LIBRARIES,
        )

        def _build_openloops_and_default_proc(force=False):
            prefix = build_openloops(force=force)
            # Default QCD NLO library (ppllj covers Drell-Yan)
            try:
                install_process_library(DEFAULT_PROCESS_LIBRARY, prefix=prefix)
                print(
                    f"[setup] Installed default QCD NLO library "
                    f"{DEFAULT_PROCESS_LIBRARY}"
                )
            except Exception as exc:
                print(
                    f"[setup] OpenLoops built; default process "
                    f"{DEFAULT_PROCESS_LIBRARY} install failed: {exc}"
                )
            # Default EW NLO libraries (Path-A finite-virtual + real emission)
            for lib in DEFAULT_EW_NLO_LIBRARIES:
                try:
                    install_process_library(lib, prefix=prefix)
                    print(f"[setup] Installed default EW NLO library {lib}")
                except Exception as exc:
                    print(
                        f"[setup] EW NLO library {lib} install failed: {exc} "
                        f"(can be installed later via `feynman install-process {lib}`)"
                    )
            return prefix

        steps.append(("OpenLoops", _build_openloops_and_default_proc))

    results: list[tuple[str, str, str]] = []
    failures = False

    for label, builder in steps:
        print(f"[setup] Building {label}...")
        try:
            built = builder(force=force)
        except Exception as exc:  # pragma: no cover - exercised via tests
            failures = True
            results.append((label, "failed", str(exc)))
            print(f"[setup] {label} failed")
        else:
            results.append((label, "ok", str(built)))
            print(f"[setup] {label} ready at {built}")

    if not include_looptools:
        results.append(("LoopTools", "skipped", "not requested"))
    if not include_openloops:
        results.append(("OpenLoops", "skipped", "not requested"))

    print("\nSetup summary:")
    for label, status, detail in results:
        print(f"  {label}: {status} - {detail}")

    if failures:
        print(
            "\nOne or more setup steps failed. "
            "You can rerun individual installers after fixing the missing toolchain."
        )
        return 1

    skipped = []
    if not include_looptools:
        skipped.append("install-looptools (numerical 1-loop)")
    if not include_openloops:
        skipped.append("install-openloops (generic NLO for arbitrary processes)")
    if skipped:
        print("\nSetup complete. Optional add-ons you skipped:")
        for s in skipped:
            print(f"  feynman {s}")
    else:
        print("\nSetup complete.")
    return 0


def _setup_wizard() -> list[str]:
    """Interactive picker for OpenLoops process packs.

    Shown only when stdin is a TTY.  Returns the resolved pack list (e.g.
    ['textbook'] or ['textbook', 'lhc-higgs']).  Empty list = no packs.
    """
    from feynman_engine.resources.openloops import PACKS, pack_summary

    print()
    print("─" * 68)
    print(" FeynmanEngine setup wizard")
    print("─" * 68)
    print()
    print("OpenLoops process libraries provide numerical |M|² for processes")
    print("without a closed-form curated formula.  Pick a pack matching your")
    print("research focus — you can install more later via")
    print("`feynman install-process <name>`.")
    print()

    # Two ways to choose: by audience (researcher type) or by theory.
    # The list is grouped with a header line so the menu reads as two
    # parallel paths to the same outcome.
    profiles = [
        # ── By audience ────────────────────────────────────────────────
        ("1", "student",       ["textbook"],
            "Textbook physics: DY, top pair, Bhabha, ee→μμ"),
        ("2", "lhc-analyst",   ["textbook", "all-lhc-nlo"],
            "LHC analyst: Higgs + top + V+jets + di-boson at NLO QCD"),
        ("3", "theorist",      ["textbook", "ee-future", "ew-only"],
            "Theorist: lepton colliders + EW NLO across the SM"),
        ("4", "ee-future",     ["textbook", "ee-future"],
            "Lepton collider: FCC-ee / ILC / muon collider"),
        # ── By theory ──────────────────────────────────────────────────
        ("5", "qed",           ["textbook"],
            "QED only — every QED process is curated; no extra OL needed"),
        ("6", "qcd",           ["textbook", "qcd-only"],
            "QCD only — every OL library with LO α_s coupling"),
        ("7", "ew",            ["textbook", "ew-only"],
            "EW only — every OL library with LO α coupling"),
        ("8", "bsm",           ["textbook"],
            "BSM only — bundled Z′+DM model is fully curated"),
        ("9", "qed-qcd",       ["textbook", "qcd-only"],
            "QED + QCD (the QCDxQED mixed sector + pure QCD)"),
        ("10", "qed-ew",       ["textbook", "ee-future", "ew-only"],
            "QED + EW (charged-lepton physics + EW NLO at lepton colliders)"),
        ("11", "qcd-ew",       ["textbook", "qcd-only", "ew-only"],
            "QCD + EW (full SM at LHC + lepton colliders)"),
        ("12", "sm",           ["textbook", "qcd-only", "ew-only"],
            "Standard Model (QCD + EW + textbook)"),
        # ── Catch-all ──────────────────────────────────────────────────
        ("13", "minimal",      [],
            "Just QGRAF + FORM + LHAPDF (no OL packs — curated only)"),
        ("14", "everything",   ["all"],
            "Everything: all 222 OL libraries (~5 GB, ~8.5 h compile)"),
    ]

    print("  ── By audience ─────────────────────────────────────────")
    last_group = "audience"
    for num, key, packs, desc in profiles:
        # Group separators: print header before transitioning.
        if num == "5" and last_group == "audience":
            print("  ── By theory ───────────────────────────────────────────")
            last_group = "theory"
        elif num == "13" and last_group == "theory":
            print("  ── Other ───────────────────────────────────────────────")
            last_group = "other"

        if packs:
            libs = []
            for p in packs:
                libs += pack_summary(p)["resolved_libraries"]
            libs = sorted(set(libs))
            from feynman_engine.resources.openloops import estimate_install
            est = estimate_install(libs)
            est_str = f"[{est['human_disk']}, {est['human_time']}]"
        else:
            est_str = "[no extra disk/time]"
        # Right-pad the key column so the disk/time estimates line up.
        print(f"  {num:>2}. {key:14s} — {desc}")
        print(f"      {est_str}")

    while True:
        try:
            choice = input(
                "Pick a profile [1-14] or type the name (default: 1=student): "
            ).strip()
        except KeyboardInterrupt:
            print("\n[setup] Aborted by user.")
            raise SystemExit(130)
        except EOFError:
            print()
            return ["textbook"]
        if choice == "":
            choice = "1"
        for num, key, packs, _desc in profiles:
            if choice == num or choice == key:
                if packs:
                    print(f"\nSelected: {key}  →  packs: {packs}")
                else:
                    print(f"\nSelected: {key}  (no OL packs)")
                return packs
        print(f"Unrecognized choice {choice!r}.  Try again.")


def _profile_to_packs(profile: str) -> list[str]:
    """Map a CLI --profile name to a list of OL packs.

    Profiles fall into three groups:

    **Audience-first** (pick what kind of physicist you are):
      - student / phd-student / phd  → textbook starter (4 libs, ~80 MB, ~8 min)
      - experimentalist / lhc-analyst → textbook + all LHC NLO QCD (~24 libs, ~720 MB)
      - theorist / theoretical       → textbook + ee-future + ew-only (~83 libs, ~1.9 GB)
      - ee-future / lepton-collider  → textbook + lepton-collider EW NLO (~9 libs, ~300 MB)

    **Theory-first** (pick what you study):
      - qed     → textbook (every QED process is curated; no extra OL libs needed)
      - qcd     → textbook + qcd-only (every OL library where the LO coupling is α_s)
      - ew      → textbook + ew-only (every OL library where the LO coupling is α)
      - bsm     → textbook (the bundled BSM model is fully curated; no OL libs)
      - qed-qcd → textbook + qcd-only (mixed QCDxQED stays in qcd-only)
      - qcd-qed → textbook + qcd-only (alias)
      - qed-ew  → textbook + ew-only + ee-future (charged-lepton physics + EW NLO)
      - ew-qed  → textbook + ew-only + ee-future (alias)
      - qcd-ew  → textbook + qcd-only + ew-only (full SM at LHC + lepton colliders)
      - sm / standard-model → textbook + qcd-only + ew-only (whole SM)

    **Catch-all**:
      - everything / all → all 222 public OL libraries (~5 GB, ~8.5 h)
      - minimal / none   → no OL packs (the existing default — curated + tabulated K only)
    """
    aliases = {
        # Audience-first
        "student":             ["textbook"],
        "phd-student":         ["textbook"],
        "phd":                 ["textbook"],
        "experimentalist":     ["textbook", "all-lhc-nlo"],
        "lhc-analyst":         ["textbook", "all-lhc-nlo"],
        "theorist":            ["textbook", "ee-future", "ew-only"],
        "theoretical":         ["textbook", "ee-future", "ew-only"],
        "ee-future":           ["textbook", "ee-future"],
        "lepton-collider":     ["textbook", "ee-future"],
        # Theory-first (single theory)
        "qed":                 ["textbook"],
        "qcd":                 ["textbook", "qcd-only"],
        "ew":                  ["textbook", "ew-only"],
        "bsm":                 ["textbook"],
        "qcd-only":            ["textbook", "qcd-only"],   # alias
        "ew-only":             ["textbook", "ew-only"],     # alias
        # Theory combinations
        "qed-qcd":             ["textbook", "qcd-only"],
        "qcd-qed":             ["textbook", "qcd-only"],
        "qed-ew":              ["textbook", "ee-future", "ew-only"],
        "ew-qed":              ["textbook", "ee-future", "ew-only"],
        "qcd-ew":              ["textbook", "qcd-only", "ew-only"],
        "ew-qcd":              ["textbook", "qcd-only", "ew-only"],
        "sm":                  ["textbook", "qcd-only", "ew-only"],
        "standard-model":      ["textbook", "qcd-only", "ew-only"],
        # Catch-all
        "everything":          ["all"],
        "all":                 ["all"],
        "minimal":             [],
        "none":                [],
    }
    return aliases.get(profile, [])


def _install_packs(packs: list[str], force: bool) -> int:
    """Install every library in the resolved pack list."""
    from feynman_engine.resources.openloops import resolve_pack, estimate_install
    from feynman_engine.openloops import install_process_library
    from feynman_engine.amplitudes.openloops_bridge import installed_processes

    libs: list[str] = []
    seen: set[str] = set()
    for pack in packs:
        for lib in resolve_pack(pack):
            if lib not in seen:
                libs.append(lib)
                seen.add(lib)

    already = set(installed_processes())
    todo = [l for l in libs if l not in already]
    if not todo:
        print("\n[install-packs] All requested libraries already installed.")
        return 0

    est = estimate_install(todo)
    print(
        f"\n[install-packs] Installing {len(todo)} OpenLoops libraries "
        f"({est['human_disk']}, {est['human_time']})..."
    )
    n_ok = 0
    for i, lib in enumerate(todo, 1):
        print(f"\n[install-packs] ({i}/{len(todo)}) {lib} ...")
        try:
            install_process_library(lib)
            n_ok += 1
        except Exception as exc:
            print(f"[install-packs] {lib} FAILED: {exc}")
    print(
        f"\n[install-packs] {n_ok}/{len(todo)} libraries installed "
        f"({len(todo) - n_ok} failed — they can be retried with "
        "`feynman install-process <name>`)."
    )
    return 0 if n_ok == len(todo) else 1


def _run_doctor() -> int:
    from feynman_engine.diagnostics import collect_diagnostics, summarize_doctor_report

    diagnostics = collect_diagnostics()
    print(summarize_doctor_report(diagnostics))
    if diagnostics["qgraf"]["available"] and diagnostics["form"]["available"]:
        return 0
    return 1


def _maybe_print_first_run_banner() -> None:
    """Show a one-time hint pointing new users at the setup wizard.

    Runs only when no top-level command was provided AND no native
    dependency has been built yet — so existing users never see it.
    """
    import os
    from pathlib import Path
    home = Path.home()
    marker = home / ".feynman" / "setup_seen"
    # Heuristic for "fresh install": no FORM, no QGRAF binaries discoverable.
    # These are the two universally-required deps (LHAPDF/OL are optional).
    have_form = bool(os.environ.get("FORM_BIN")) or any(
        (Path(p) / "form").exists()
        for p in os.environ.get("PATH", "").split(os.pathsep) if p
    )
    have_qgraf = bool(os.environ.get("QGRAF_BIN")) or any(
        (Path(p) / "qgraf").exists()
        for p in os.environ.get("PATH", "").split(os.pathsep) if p
    )
    if marker.exists() or (have_form and have_qgraf):
        return
    print(
        "─" * 68
        + "\n FeynmanEngine — first run\n"
        + "─" * 68
        + "\n No native dependencies (QGRAF, FORM) detected on this machine."
        + "\n Run `feynman setup` to launch the interactive setup wizard:"
        + "\n   • picks a process-library pack matching your physics focus"
        + "\n   • shows honest disk + time estimates before committing"
        + "\n   • or use `--profile minimal` to skip OpenLoops packs"
        + "\n " + "─" * 67
        + "\n"
    )
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except Exception:
        pass


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="feynman",
        description="FeynmanEngine — Feynman diagram generator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── serve ──────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    # ── generate ───────────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser("generate", help="Generate diagrams from the command line")
    gen_parser.add_argument("process", help='Scattering process, e.g. "e+ e- -> mu+ mu-"')
    gen_parser.add_argument("--theory", default="QED")
    gen_parser.add_argument("--loops", type=int, default=0)
    gen_parser.add_argument("--format", dest="output_format", default="tikz",
                            choices=["tikz", "svg", "png"])
    gen_parser.add_argument("--output-dir", default=".", help="Directory to write output files")

    # ── install-qgraf ─────────────────────────────────────────────────────
    qgraf_parser = subparsers.add_parser(
        "install-qgraf",
        help="Compile QGRAF from the bundled source archive",
    )
    qgraf_parser.add_argument(
        "--target",
        default=None,
        help="Optional output path for the compiled qgraf binary",
    )
    qgraf_parser.add_argument("--force", action="store_true")

    # ── install-looptools ─────────────────────────────────────────────────
    lt_parser = subparsers.add_parser(
        "install-looptools",
        help="Build LoopTools shared library from the bundled source archive",
    )
    lt_parser.add_argument(
        "--target",
        default=None,
        help="Optional output path for the compiled shared library",
    )
    lt_parser.add_argument("--force", action="store_true")

    # ── install-form ───────────────────────────────────────────────────
    form_parser = subparsers.add_parser(
        "install-form",
        help="Build FORM symbolic algebra tool from the bundled source archive",
    )
    form_parser.add_argument(
        "--target",
        default=None,
        help="Optional output path for the compiled form binary",
    )
    form_parser.add_argument("--force", action="store_true")

    # ── install-lhapdf ────────────────────────────────────────────────────
    lhapdf_parser = subparsers.add_parser(
        "install-lhapdf",
        help="Build LHAPDF (PDF library) from the bundled source archive",
    )
    lhapdf_parser.add_argument(
        "--target",
        default=None,
        help="Install prefix (defaults to /tmp/lhapdf-install or ~/.local/lhapdf)",
    )
    lhapdf_parser.add_argument("--force", action="store_true")
    lhapdf_parser.add_argument(
        "--no-set",
        action="store_true",
        help="Skip auto-installing the CT18LO PDF set after building LHAPDF",
    )

    # ── install-pdf-set ───────────────────────────────────────────────────
    pdf_parser = subparsers.add_parser(
        "install-pdf-set",
        help="Download and install an LHAPDF PDF set (e.g. CT18LO, NNPDF40_lo_as_01180)",
    )
    pdf_parser.add_argument(
        "set_name",
        nargs="?",
        default="CT18LO",
        help="LHAPDF set name (default: CT18LO)",
    )
    pdf_parser.add_argument(
        "--prefix",
        default=None,
        help="LHAPDF install prefix (uses the auto-discovered location by default)",
    )

    # ── install-openloops ─────────────────────────────────────────────────
    ol_parser = subparsers.add_parser(
        "install-openloops",
        help="Build OpenLoops 2 (one-loop amplitude generator) from the bundled source",
    )
    ol_parser.add_argument(
        "--target",
        default=None,
        help="Install prefix (defaults to /opt/openloops, /tmp/openloops-install, or ~/.local/openloops)",
    )
    ol_parser.add_argument("--force", action="store_true")
    ol_parser.add_argument(
        "--no-process",
        action="store_true",
        help="Skip auto-installing the default process libraries (ppllj + EW NLO libs)",
    )
    ol_parser.add_argument(
        "--no-ew-libs",
        action="store_true",
        help=(
            "Skip auto-installing the default EW NLO libraries (eell_ew, eella_ew). "
            "Without these, EW NLO requests fall back to the analytic Sudakov LL+NLL "
            "approximation. Useful in CI to save ~2-3 minutes."
        ),
    )

    # ── install-process ───────────────────────────────────────────────────
    proc_parser = subparsers.add_parser(
        "install-process",
        help="Download and compile an OpenLoops process library (e.g. ppllj, pptt, pph, pphjj)",
    )
    proc_parser.add_argument(
        "process",
        nargs="?",
        default="ppllj",
        help=(
            "OpenLoops process library name (default: ppllj).  Use 'all-lhc' "
            "for the full LHC pack, 'all-ee' for the lepton-collider pack, "
            "or 'all' for everything (long install — see below)."
        ),
    )
    proc_parser.add_argument(
        "--prefix",
        default=None,
        help="OpenLoops install prefix (uses the auto-discovered location by default)",
    )
    proc_parser.add_argument(
        "--all", action="store_true",
        help=(
            "Install ALL libraries in COMMON_PROCESS_LIBRARIES + DEFAULT_EW_NLO_LIBRARIES "
            "(~12 libraries, ~30 min compile, ~3 GB disk).  Equivalent to passing "
            "process='all'."
        ),
    )
    proc_parser.add_argument(
        "--lhc-pack", action="store_true",
        help=(
            "Install the LHC-process pack: ppllj, pptt, ppttj, pphtt, pph, ppvv, "
            "pphjj, pphh, ppvvj, ppvjj.  ~25 min, ~2.5 GB."
        ),
    )
    proc_parser.add_argument(
        "--ee-pack", action="store_true",
        help=(
            "Install the lepton-collider pack: eell_ew, eella_ew, eett_ew, eevv_ew. "
            "~5 min, ~500 MB."
        ),
    )

    # ── setup / install-all ───────────────────────────────────────────────
    setup_parser = subparsers.add_parser(
        "setup",
        aliases=["install-all"],
        help="Build the bundled native dependencies (interactive wizard by default)",
    )
    setup_parser.add_argument(
        "--non-interactive", action="store_true",
        help="Skip the interactive wizard; install only the default starter (no OL packs).",
    )
    setup_parser.add_argument(
        "--profile",
        choices=[
            # Audience-first
            "student", "phd-student", "phd",
            "experimentalist", "lhc-analyst",
            "theorist", "theoretical",
            "ee-future", "lepton-collider",
            # Theory-first (single)
            "qed", "qcd", "ew", "bsm",
            "qcd-only", "ew-only",
            # Theory combinations
            "qed-qcd", "qcd-qed",
            "qed-ew", "ew-qed",
            "qcd-ew", "ew-qcd",
            "sm", "standard-model",
            # Catch-all
            "everything", "all",
            "minimal", "none",
        ],
        default=None,
        help=(
            "Skip the interactive wizard and install a named profile.  "
            "Audience-first: student / experimentalist / theorist / ee-future.  "
            "Theory-first: qed / qcd / ew / bsm / qed-qcd / qed-ew / qcd-ew / sm.  "
            "student=textbook pack; lhc-analyst=textbook+all-lhc-nlo; "
            "ee-future=textbook+ee-future; qcd-only/ew-only filter by theory; "
            "minimal=no OL packs."
        ),
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild binaries/libraries even if the target already exists",
    )
    setup_parser.add_argument(
        "--skip-looptools",
        action="store_true",
        help="Skip LoopTools (numerical 1-loop scalar integrals)",
    )
    setup_parser.add_argument(
        "--skip-openloops",
        action="store_true",
        help=(
            "Skip OpenLoops 2 (saves ~5-10 min; without it, generic NLO "
            "QCD virtuals for unregistered processes return HTTP 422 — "
            "tabulated K-factors for the major LHC channels still work)"
        ),
    )
    # Deprecated opt-in flags (kept for backward compatibility, no-op now).
    setup_parser.add_argument(
        "--with-lhapdf", action="store_true", help=argparse.SUPPRESS,
    )
    setup_parser.add_argument(
        "--with-openloops", action="store_true", help=argparse.SUPPRESS,
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect which native dependencies are installed and where they were found",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw diagnostics as JSON",
    )

    args = parser.parse_args()

    # First-run banner: shown only the very first time a user invokes the
    # CLI, when no native deps are detected.  Skip for `setup` (the user
    # is already running it) and `doctor` (which has its own report).
    if args.command not in ("setup", "install-all", "doctor"):
        _maybe_print_first_run_banner()

    if args.command == "serve":
        import uvicorn
        uvicorn.run(
            "feynman_engine.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "generate":
        from feynman_engine.engine import FeynmanEngine

        engine = FeynmanEngine()
        print(f"Generating: {args.process} [{args.theory}, loops={args.loops}]")
        result = engine.generate(
            process=args.process,
            theory=args.theory,
            loops=args.loops,
            output_format=args.output_format,
        )

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for d in result.diagrams:
            stem = f"diagram_{d.id}_{d.topology or 'unknown'}"
            tikz = result.tikz_code.get(d.id)
            if tikz:
                tex_path = out_dir / f"{stem}.tex"
                tex_path.write_text(tikz)
                print(f"  Wrote {tex_path}")

            image = result.images.get(d.id)
            if image:
                ext = args.output_format
                img_path = out_dir / f"{stem}.{ext}"
                img_path.write_bytes(image)
                print(f"  Wrote {img_path}")

        print(f"\nSummary: {json.dumps(result.summary, indent=2)}")

    elif args.command == "install-qgraf":
        from feynman_engine.qgraf import build_qgraf

        target = Path(args.target).expanduser() if args.target else None
        built = build_qgraf(target=target, force=args.force)
        print(f"Built QGRAF at {built}")

    elif args.command == "install-looptools":
        from feynman_engine.looptools import build_looptools

        target = Path(args.target).expanduser() if args.target else None
        built = build_looptools(target=target, force=args.force)
        print(f"Built LoopTools at {built}")

    elif args.command == "install-form":
        from feynman_engine.form import build_form

        target = Path(args.target).expanduser() if args.target else None
        built = build_form(target=target, force=args.force)
        print(f"Built FORM at {built}")

    elif args.command == "install-lhapdf":
        from feynman_engine.lhapdf import build_lhapdf, install_pdf_set, DEFAULT_PDF_SET

        target = Path(args.target).expanduser() if args.target else None
        prefix = build_lhapdf(target=target, force=args.force)
        print(f"Built LHAPDF at {prefix}")
        if not args.no_set:
            try:
                set_dir = install_pdf_set(DEFAULT_PDF_SET, prefix=prefix)
                print(f"Installed default PDF set: {set_dir}")
            except Exception as exc:
                print(f"⚠ Default PDF set install failed: {exc}")
                print("  Run `feynman install-pdf-set CT18LO` manually after fixing the issue.")

    elif args.command == "install-pdf-set":
        from feynman_engine.lhapdf import install_pdf_set

        set_dir = install_pdf_set(args.set_name, prefix=args.prefix)
        print(f"Installed PDF set {args.set_name} at {set_dir}")

    elif args.command == "install-openloops":
        from feynman_engine.openloops import (
            build_openloops, install_process_library,
            DEFAULT_PROCESS_LIBRARY, DEFAULT_EW_NLO_LIBRARIES,
        )

        target = Path(args.target).expanduser() if args.target else None
        prefix = build_openloops(target=target, force=args.force)
        print(f"Built OpenLoops at {prefix}")
        if not args.no_process:
            # Default QCD NLO library
            try:
                proclib = install_process_library(DEFAULT_PROCESS_LIBRARY, prefix=prefix)
                print(f"Installed default QCD NLO library {DEFAULT_PROCESS_LIBRARY} in {proclib}")
            except Exception as exc:
                print(f"⚠ Default QCD process library install failed: {exc}")
                print(
                    f"  Run `feynman install-process {DEFAULT_PROCESS_LIBRARY}` "
                    "manually after fixing the issue."
                )
            # Default EW NLO libraries (unblocks Path-A EW NLO out of the box)
            if not args.no_ew_libs:
                for lib in DEFAULT_EW_NLO_LIBRARIES:
                    try:
                        install_process_library(lib, prefix=prefix)
                        print(f"Installed default EW NLO library {lib}")
                    except Exception as exc:
                        print(f"⚠ EW NLO library {lib} install failed: {exc}")
                        print(
                            f"  Run `feynman install-process {lib}` "
                            "manually after fixing the issue."
                        )

    elif args.command == "install-process":
        from feynman_engine.openloops import (
            install_process_library, COMMON_PROCESS_LIBRARIES,
            DEFAULT_EW_NLO_LIBRARIES,
        )

        # Resolve process pack selection
        to_install: list[str] = []
        if args.all or args.process in {"all", "all-libraries"}:
            to_install = list(COMMON_PROCESS_LIBRARIES) + list(DEFAULT_EW_NLO_LIBRARIES)
        elif args.lhc_pack or args.process == "all-lhc":
            to_install = list(COMMON_PROCESS_LIBRARIES)
        elif args.ee_pack or args.process == "all-ee":
            to_install = list(DEFAULT_EW_NLO_LIBRARIES) + ["eett_ew", "eevv_ew"]
        else:
            to_install = [args.process]

        if len(to_install) > 1:
            print(
                f"[install-process] Installing {len(to_install)} libraries: "
                f"{', '.join(to_install)}"
            )
            print("  Estimated time: ~3-5 min per library, ~30 min total for the full set.")
            print("  Estimated disk: ~50-200 MB per library, ~3 GB for the full set.\n")

        n_ok = 0
        for proc in to_install:
            try:
                proclib = install_process_library(proc, prefix=args.prefix)
                print(f"  ✓ Installed {proc} in {proclib}")
                n_ok += 1
            except Exception as exc:
                print(f"  ✗ {proc} failed: {exc}")

        if len(to_install) > 1:
            print(f"\n[install-process] {n_ok}/{len(to_install)} libraries installed.")

    elif args.command in {"setup", "install-all"}:
        # Three modes (profile wins over --non-interactive when both are given):
        #   1. --profile <name>:  scripted install of a named profile
        #   2. --non-interactive: bare deps only, no OL packs
        #   3. default:           interactive wizard (when on a TTY)
        if args.profile:
            chosen_packs = _profile_to_packs(args.profile)
        elif args.non_interactive:
            chosen_packs: list[str] = []
        elif sys.stdin.isatty():
            chosen_packs = _setup_wizard()
        else:
            chosen_packs = []  # non-TTY (CI, pipe) → no interactive prompts

        code = _run_setup(
            force=args.force,
            include_looptools=not args.skip_looptools,
            include_openloops=(not args.skip_openloops) or bool(chosen_packs),
        )
        if code == 0 and chosen_packs:
            code = _install_packs(chosen_packs, force=args.force)
        raise SystemExit(code)

    elif args.command == "doctor":
        if args.json:
            from feynman_engine.diagnostics import collect_diagnostics

            print(json.dumps(collect_diagnostics(), indent=2))
            raise SystemExit(0)
        raise SystemExit(_run_doctor())


if __name__ == "__main__":
    main()
