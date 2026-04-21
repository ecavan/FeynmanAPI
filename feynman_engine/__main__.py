"""CLI entry point for FeynmanEngine."""
import sys


def _run_setup(force: bool, include_looptools: bool) -> int:
    from feynman_engine.form import build_form
    from feynman_engine.looptools import build_looptools
    from feynman_engine.qgraf import build_qgraf

    steps = [
        ("QGRAF", build_qgraf),
        ("FORM", build_form),
    ]
    if include_looptools:
        steps.append(("LoopTools", build_looptools))

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

    print("\nSetup summary:")
    for label, status, detail in results:
        print(f"  {label}: {status} - {detail}")

    if failures:
        print(
            "\nOne or more setup steps failed. "
            "You can rerun individual installers after fixing the missing toolchain."
        )
        return 1

    if not include_looptools:
        print(
            "\nSetup complete. "
            "Run `feynman install-looptools` later if you want numerical 1-loop evaluation."
        )
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

    elif args.command in {"setup", "install-all"}:
        code = _run_setup(force=args.force, include_looptools=not args.skip_looptools)
        raise SystemExit(code)

    elif args.command == "doctor":
        if args.json:
            from feynman_engine.diagnostics import collect_diagnostics

            print(json.dumps(collect_diagnostics(), indent=2))
            raise SystemExit(0)
        raise SystemExit(_run_doctor())


if __name__ == "__main__":
    main()
