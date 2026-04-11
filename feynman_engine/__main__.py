"""CLI entry point for FeynmanEngine."""
import sys


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


if __name__ == "__main__":
    main()
