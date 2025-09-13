"""Command line interface for Little Garden HPLC/DAD toolkit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .io import load_any
from .preprocess import apply_pipeline, sanitize_spectrum
from .sticks import pick_sticks
from .features import compute_features
from .library import add_replicate, build_index
from .matching import score
from .sim import sim_traces


def fp_make(input_folder: Path, out_folder: Path) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)
    for lam_file in input_folder.glob('*_lam.npy'):
        lam, spec, meta = load_any(lam_file)
        y = spec[0] if spec.ndim > 1 else spec
        lam, y, notes = sanitize_spectrum(lam, y)
        y_hat = apply_pipeline(lam, y, [{'op': 'snv'}, {'op': 'savgol', 'win': 7, 'poly': 2}])
        sticks = pick_sticks(lam, y_hat)
        fp = compute_features(lam, y_hat, sticks)
        fp['meta'] = meta
        fp['warnings'].extend(notes)
        base = lam_file.stem.replace('_lam', '')
        out_json = out_folder / f"{base}.pms.json"
        add_replicate(fp, out_json)
        np.savez(out_folder / f"{base}.npz", lam=lam, spec=y_hat)
        if fp['warnings']:
            print(f"{base}: {'; '.join(fp['warnings'])}")


def lib_build(input_folder: Path, out_file: Path) -> None:
    index = build_index(input_folder)
    out_file.write_text(json.dumps(index, indent=2))


def match_sample(sample_file: Path, lib_file: Path, top: int, json_out: Path | None) -> None:
    lam, spec, _ = load_any(sample_file)
    y = spec[0] if spec.ndim > 1 else spec
    lam, y, _ = sanitize_spectrum(lam, y)
    y_hat = apply_pipeline(lam, y, [{'op': 'snv'}, {'op': 'savgol', 'win': 7, 'poly': 2}])
    sticks = pick_sticks(lam, y_hat)
    fp = compute_features(lam, y_hat, sticks)
    library = json.loads(Path(lib_file).read_text())
    rows = []
    for name, entry in library.items():
        lib_entry = {
            'sticks': entry.get('sticks', []),
            'ratios': entry.get('ratios_mean', []),
            'bandpower': entry.get('bandpower_mean', []),
            'hash': entry.get('hash'),
        }
        sc = score(fp, lib_entry)
        rows.append({'name': name, **sc})
    rows.sort(key=lambda r: r['S'], reverse=True)
    for r in rows[:top]:
        print(f"{r['name']}\t{r['S']:.3f}")
    if json_out:
        json_out.write_text(json.dumps(rows[:top], indent=2))


def sim_export(in_file: Path, out_file: Path, channels: list[float]) -> None:
    lam, spec, _ = load_any(in_file)
    if spec.ndim != 2:
        raise ValueError('SIM export requires time-resolved spectra')
    traces = sim_traces(lam, spec, channels)
    if not traces:
        raise ValueError('no channels found')
    header = ','.join(traces.keys())
    arr = np.column_stack([traces[k] for k in traces])
    np.savetxt(out_file, arr, delimiter=',', header=header, comments='')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='lgfp')
    sub = parser.add_subparsers(dest='cmd', required=True)

    fp_parser = sub.add_parser('fp', help='fingerprint utilities')
    fp_sub = fp_parser.add_subparsers(dest='subcmd', required=True)
    make_p = fp_sub.add_parser('make', help='create fingerprints')
    make_p.add_argument('--in', dest='input', type=Path, required=True)
    make_p.add_argument('--out', type=Path, required=True)

    lib_parser = sub.add_parser('lib', help='library tools')
    lib_sub = lib_parser.add_subparsers(dest='subcmd', required=True)
    build_p = lib_sub.add_parser('build', help='build library index')
    build_p.add_argument('--in', dest='input', type=Path, required=True)
    build_p.add_argument('--out', type=Path, required=True)

    match_p = sub.add_parser('match', help='match sample against library')
    match_p.add_argument('--sample', type=Path, required=True)
    match_p.add_argument('--lib', type=Path, required=True)
    match_p.add_argument('--top', type=int, default=10)
    match_p.add_argument('--json', type=Path)

    sim_parser = sub.add_parser('sim', help='SIM utilities')
    sim_sub = sim_parser.add_subparsers(dest='subcmd', required=True)
    export_p = sim_sub.add_parser('export', help='export SIM traces')
    export_p.add_argument('--in', dest='input', type=Path, required=True)
    export_p.add_argument('--out', type=Path, required=True)
    export_p.add_argument('--channels', type=str, required=True,
                         help='comma separated wavelengths')

    args = parser.parse_args(argv)
    try:
        if args.cmd == 'fp' and args.subcmd == 'make':
            fp_make(args.input, args.out)
        elif args.cmd == 'lib' and args.subcmd == 'build':
            lib_build(args.input, args.out)
        elif args.cmd == 'match':
            match_sample(args.sample, args.lib, args.top, args.json)
        elif args.cmd == 'sim' and args.subcmd == 'export':
            channels = [float(c) for c in args.channels.split(',') if c]
            sim_export(args.input, args.out, channels)
        return 0
    except Exception as exc:  # pragma: no cover - CLIs
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
