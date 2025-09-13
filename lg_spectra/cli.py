"""Command line interface for Little Garden HPLC/DAD toolkit."""

from __future__ import annotations

import argparse
from pathlib import Path


from .io import load_any
from .preprocess import apply_pipeline
from .sticks import pick_sticks
from .features import compute_features
from .library import add_replicate, build_index


def fp_make(input_folder: Path, out_folder: Path) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    for lam_file in input_folder.glob("*_lam.npy"):
        lam, spec, meta = load_any(lam_file)
        y = spec[0] if spec.ndim > 1 else spec
        y_hat = apply_pipeline(
            lam, y, [{"op": "snv"}, {"op": "savgol", "win": 7, "poly": 2}]
        )
        sticks = pick_sticks(lam, y_hat)
        fp = compute_features(lam, y_hat, sticks)
        fp["meta"] = meta
        out_path = out_folder / (lam_file.stem.replace("_lam", "") + ".pms.json")
        add_replicate(fp, out_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lgfp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fp_parser = sub.add_parser("fp", help="fingerprint utilities")
    fp_sub = fp_parser.add_subparsers(dest="subcmd", required=True)
    make_p = fp_sub.add_parser("make")
    make_p.add_argument("input_folder", type=Path)
    make_p.add_argument("--out", type=Path, required=True)

    lib_parser = sub.add_parser("lib", help="library tools")
    lib_sub = lib_parser.add_subparsers(dest="subcmd", required=True)
    build_p = lib_sub.add_parser("build")
    build_p.add_argument("db_folder", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "fp" and args.subcmd == "make":
        fp_make(args.input_folder, args.out)
    elif args.cmd == "lib" and args.subcmd == "build":
        build_index(args.db_folder)


if __name__ == "__main__":  # pragma: no cover
    main()
