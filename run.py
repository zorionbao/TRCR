"""Run the core analyses, numerical benchmark, and reusable panel examples."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def configure_runtime(output_dir: Path) -> None:
    """Keep execution files together and use deterministic numerical threading."""
    runtime = output_dir / ".runtime"
    temporary = runtime / "tmp"
    matplotlib = runtime / "matplotlib"
    temporary.mkdir(parents=True, exist_ok=True)
    matplotlib.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
        TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary),
        MPLCONFIGDIR=str(matplotlib), OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    actions = command.add_subparsers(dest="command", required=True)
    analysis = actions.add_parser("analyze", help="Fit cohort models and prepare reusable panel-example inputs")
    analysis.add_argument("--source-dir", type=Path, required=True,
                          help="Directory containing hrs.csv, elsa.csv, and charls.csv")
    benchmark = actions.add_parser("benchmark", help="Run the 600-replicate MOSAIC benchmark")
    planning = actions.add_parser("rift", help="Compute resource frontiers from estimated joint states")
    planning.add_argument("--input-dir", type=Path, required=True,
                          help="Directory containing joint_states.csv")
    figures = actions.add_parser("figures", help="Draw 32 reusable panel examples and four example assemblies")
    figures.add_argument("--input-dir", type=Path, required=True,
                         help="Directory containing the four Figure*.csv input tables")
    figures.add_argument("--font-dir", type=Path,
                         help="Directory containing Liberation Sans font files")
    checks = actions.add_parser("test", help="Run numerical and interface tests")
    for action in (analysis, benchmark, planning, figures, checks):
        action.add_argument("--output-dir", type=Path, default=Path("outputs"),
                            help="Execution output directory (default: outputs)")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "analyze":
        for filename in ("hrs.csv", "elsa.csv", "charls.csv"):
            if not (args.source_dir / filename).is_file():
                raise FileNotFoundError(f"Required cohort input: {args.source_dir / filename}")
    if args.command in ("figures", "rift") and not args.input_dir.is_dir():
        raise FileNotFoundError(f"Required input directory: {args.input_dir}")
    configure_runtime(args.output_dir)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.command == "analyze":
            from core import harmonize, clinical, domains, mosaic, turnover, benchmark, rift, figure_inputs
            stages = (
                ("Cohort harmonization", lambda: harmonize.run(args.source_dir, args.output_dir)),
                ("Cross-fitted functional contrasts", lambda: clinical.run(args.source_dir, args.output_dir)),
                ("Task-specific contrasts", lambda: domains.run(args.source_dir, args.output_dir)),
                ("MOSAIC joint-state decomposition", lambda: mosaic.run(args.output_dir)),
                ("Observed task turnover", lambda: turnover.run(args.output_dir)),
                ("MOSAIC calibration benchmark", lambda: benchmark.run(args.output_dir)),
                ("RIFT resource frontiers", lambda: rift.run(args.output_dir)),
                ("Reusable panel-example input tables", lambda: figure_inputs.run(args.output_dir)),
            )
            for label, execute in stages:
                print(label, flush=True)
                execute()
        elif args.command == "benchmark":
            from core import benchmark
            benchmark.run(args.output_dir)
        elif args.command == "rift":
            from core import rift
            rift.run(args.output_dir, input_dir=args.input_dir)
        elif args.command == "figures":
            from figures.assemble import run
            run(args.input_dir, args.output_dir, font_dir=args.font_dir)
        elif args.command == "test":
            import unittest
            suite = unittest.defaultTestLoader.discover(
                str(Path(__file__).parent / "tests"), pattern="test_*.py")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
