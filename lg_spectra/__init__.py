"""Little Garden HPLC/DAD spectral analysis toolkit."""

from .io import load_any
from .preprocess import apply_pipeline
from .sticks import Stick, find_peaks, to_sticks, pick_sticks
from .features import compute_features, SCHEMA
from .vectorize import sticks_to_vector
from .matching import score, match_spectrum
from .library import Library, add_replicate, build_index
from .sim import sim_traces

__all__ = [
    'load_any', 'apply_pipeline', 'Stick', 'find_peaks', 'to_sticks',
    'pick_sticks', 'compute_features', 'SCHEMA', 'sticks_to_vector',
    'score', 'match_spectrum', 'Library', 'add_replicate', 'build_index',
    'sim_traces'
]
