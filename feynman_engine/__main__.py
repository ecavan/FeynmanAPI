"""CLI entry point for FeynmanEngine."""
import sys


def _run_setup(
    force: bool,
    include_looptools: bool,
    include_lhapdf: bool,
    include_openloops: bool = False,
) -> int:
    from feynman_engine.form import build_form
    from feynman_engine.looptools import build_looptools
    from feynman_engine.qgraf import build_qgraf

    steps = [
        ("QGRAF", build_qgraf),
        ("FORM", build_form),
    ]
    if include_looptools:
        steps.append(("LoopTools", build_looptools))
    if include_lhapdf:
        from feynman_engine.lhapdf import build_lhapdf, install_pdf_set, DEFAULT_PDF_SET

        def _build_lhapdf_and_default_set(force=False):
            prefix = build_lhapdf(force=force)
            try:
                install_pdf_set(DEFAULT_PDF_SET, prefix=prefix)
            except Exception as exc:
                # Set install fails (no internet, etc.) shouldn't fail the build
                print(f"[setup] LHAPDF built; default set {DEFAULT_PDF_SET} install failed: {exc}")
            return prefix

        steps.append(("LHAPDF", _build_lhapdf_and_default_set))
    if include_openloops:
        from feynman_engine.openloops import (
            build_openloops, install_process_library, DEFAULT_PROCESS_LIBRARY,
        )

        def _build_openloops_and_default_proc(force=False):
            prefix = build_openloops(force=force)
            try:
                install_process_library(DEFAULT_PROCESS_LIBRARY, prefix=prefix)
            except Exception as exc:
                print(
                    f"[setup] OpenLoops built; default process "
                    f"{DEFAULT_PROCESS_LIBRARY} install failed: {exc}"
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
    if not include_lhapdf:
        results.append(("LHAPDF", "skipped", "not requested"))
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
    if not include_lhapdf:
        skipped.append("install-lhapdf (precision PDFs for hadron-collider σ)")
    if not include_openloops:
        skipped.append("install-openloops (generic NLO for arbitrary processes)")
    if skipped:
        print("\nSetup complete. Optional add-ons you skipped:")
        for s in skipped:
            print(f"  feynman {s}")
    else:
        print("\nSetup complete.")
    return 0


def _run_doctor() -> int:
    from feynman_engine.diagnostics import collect_diagnostics, summarize_doctor_report

    diagnostics = collect_diagnostics()
    print(summarize_doctor_report(diagnostics))
    if diagnostics["qgraf"]["available"] and diagnostics["form"]["available"]:
        return 0
    return 1


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
        help="Skip auto-installing the default ppllj process library",
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
        help="OpenLoops process library name (default: ppllj)",
    )
    proc_parser.add_argument(
        "--prefix",
        default=None,
        help="OpenLoops install prefix (uses the auto-discovered location by default)",
    )

    # ── setup / install-all ───────────────────────────────────────────────
    setup_parser = subparsers.add_parser(
        "setup",
        aliases=["install-all"],
        help="Build the bundled native dependencies in one command",
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild binaries/libraries even if the target already exists",
    )
    setup_parser.add_argument(
        "--skip-looptools",
        action="store_true",
        help="Skip LoopTools if you only want the recommended QGRAF + FORM setup",
    )
    setup_parser.add_argument(
        "--with-lhapdf",
        action="store_true",
        help=(
            "Also build LHAPDF and install the CT18LO PDF set "
            "(adds ~5 minutes to setup; recommended for hadron-collider physics)"
        ),
    )
    setup_parser.add_argument(
        "--with-openloops",
        action="store_true",
        help=(
            "Also build OpenLoops 2 and install the default ppllj process library "
            "(adds ~5-10 minutes to setup; enables generic NLO for arbitrary processes)"
        ),
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
            build_openloops, install_process_library, DEFAULT_PROCESS_LIBRARY,
        )

        target = Path(args.target).expanduser() if args.target else None
        prefix = build_openloops(target=target, force=args.force)
        print(f"Built OpenLoops at {prefix}")
        if not args.no_process:
            try:
                proclib = install_process_library(DEFAULT_PROCESS_LIBRARY, prefix=prefix)
                print(f"Installed default process library {DEFAULT_PROCESS_LIBRARY} in {proclib}")
            except Exception as exc:
                print(f"⚠ Default process library install failed: {exc}")
                print(
                    f"  Run `feynman install-process {DEFAULT_PROCESS_LIBRARY}` "
                    "manually after fixing the issue."
                )

    elif args.command == "install-process":
        from feynman_engine.openloops import install_process_library

        proclib = install_process_library(args.process, prefix=args.prefix)
        print(f"Installed OpenLoops process {args.process} in {proclib}")

    elif args.command in {"setup", "install-all"}:
        code = _run_setup(
            force=args.force,
            include_looptools=not args.skip_looptools,
            include_lhapdf=args.with_lhapdf,
            include_openloops=args.with_openloops,
        )
        raise SystemExit(code)

    elif args.command == "doctor":
        if args.json:
            from feynman_engine.diagnostics import collect_diagnostics

            print(json.dumps(collect_diagnostics(), indent=2))
            raise SystemExit(0)
        raise SystemExit(_run_doctor())


if __name__ == "__main__":
    main()
