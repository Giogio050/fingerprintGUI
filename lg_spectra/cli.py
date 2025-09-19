"""Command line interface for Little Garden HPLC/DAD toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .io import load_any, load_folder
from .library import add_replicate, build_index
from .matching import score
from .pipeline import run_pipelines
from .sim import sim_traces


def _save_fingerprint(out_dir: Path, base_name: str, fp: Dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_name}.pms.json"
    add_replicate(fp, out_path)
    return out_path


def fp_make(input_folder: Path, out_folder: Path, preset: str, top: int, max_methods: int) -> List[Path]:
    records = load_folder(input_folder)
    outputs: List[Path] = []
    for record in records:
        lam = record["wavelength"]
        spec = record["spectrum"]
        meta = record.get("meta", {})
        results = run_pipelines(lam, spec, preset=preset, max_methods=max_methods, top_n=top)
        if not results:
            continue
        best = results[0]
        fingerprint = dict(best.features)
        fingerprint["meta"] = meta | {"pipeline": best.name}
        fingerprint["diagnostics"] = best.diagnostics
        base = Path(meta.get("filename", "sample")).stem
        outputs.append(_save_fingerprint(out_folder, base, fingerprint))
    return outputs


def id_match(sample_path: Path, db_folder: Path, preset: str, max_methods: int, top: int) -> List[Dict[str, float]]:
    lam, spec, meta = load_any(sample_path)
    results = run_pipelines(lam, spec, preset=preset, max_methods=max_methods, top_n=1)
    if not results:
        raise RuntimeError("No pipelines produced a fingerprint")
    sample_fp = dict(results[0].features)
    sample_fp["meta"] = meta | {"pipeline": results[0].name}

    index_file = db_folder / "library_index.json"
    if index_file.exists():
        library = json.loads(index_file.read_text())
    else:
        library = build_index(db_folder)

    matches = []
    for name, entry in library.items():
        sc = score(sample_fp, entry)
        matches.append((name, sc))
    matches.sort(key=lambda x: x[1]["S"], reverse=True)
    for name, sc in matches[:top]:
        print(f"{name:20s} S={sc['S']:.3f} Cos={sc['S_cos']:.3f} Ratio={sc['S_ratio']:.3f}")
    return [dict(name=name, **sc) for name, sc in matches[:top]]


def sim_export(sample_path: Path, library_index: Path, out_file: Path, band: int) -> None:
    lam, spec, _ = load_any(sample_path)
    if spec.ndim != 2:
        raise ValueError("SIM export requires time-resolved spectra (frames x wavelength)")
    library = json.loads(library_index.read_text())
    target_lambdas = []
    for entry in library.values():
        sticks = entry.get("sticks", [])
        if sticks:
            target_lambdas.extend(stick["lambda_nm"] for stick in sticks[:3])
    traces = sim_traces(lam, spec, target_lambdas, band=band)
    out_file.write_text(json.dumps(traces, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lgfp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fp_parser = sub.add_parser("fp", help="fingerprint utilities")
    fp_sub = fp_parser.add_subparsers(dest="subcmd", required=True)
    make_p = fp_sub.add_parser("make")
    make_p.add_argument("input_folder", type=Path)
    make_p.add_argument("--out", type=Path, required=True)
    make_p.add_argument("--preset", default="NoiseMax")
    make_p.add_argument("--top", type=int, default=3)
    make_p.add_argument("--max-methods", type=int, default=90)

    lib_parser = sub.add_parser("lib", help="library tools")
    lib_sub = lib_parser.add_subparsers(dest="subcmd", required=True)
    build_p = lib_sub.add_parser("build")
    build_p.add_argument("db_folder", type=Path)
    build_p.add_argument("--out", type=Path)

    id_parser = sub.add_parser("id", help="identification")
    id_parser.add_argument("sample", type=Path)
    id_parser.add_argument("--db", type=Path, required=True)
    id_parser.add_argument("--preset", default="NoiseMax")
    id_parser.add_argument("--max-methods", type=int, default=90)
    id_parser.add_argument("--top", type=int, default=10)

    sim_parser = sub.add_parser("sim", help="SIM trace export")
    sim_parser.add_argument("sample", type=Path)
    sim_parser.add_argument("--library", type=Path, required=True)
    sim_parser.add_argument("--out", type=Path, required=True)
    sim_parser.add_argument("--band", type=int, default=2)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "fp" and args.subcmd == "make":
        outputs = fp_make(args.input_folder, args.out, args.preset, args.top, args.max_methods)
        for out in outputs:
            print(out)
    elif args.cmd == "lib" and args.subcmd == "build":
        index = build_index(args.db_folder, args.out)
        print(f"Indexed {len(index)} analytes")
    elif args.cmd == "id":
        id_match(args.sample, args.db, args.preset, args.max_methods, args.top)
    elif args.cmd == "sim":
        sim_export(args.sample, args.library, args.out, args.band)


if __name__ == "__main__":  # pragma: no cover
    main()
