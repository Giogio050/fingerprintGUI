"""Little Garden HPLC/DAD spectral analysis toolkit."""

from .io import load_any, load_folder, save_spec
from .preprocess import apply_pipeline
from .sticks import pick_sticks, Stick
from .features import compute_features
from .vectorize import sticks_to_vector
from .matching import score
from .library import add_replicate, build_index
from .sim import sim_traces

__all__ = [
    "load_any",
    "load_folder",
    "save_spec",
    "apply_pipeline",
    "pick_sticks",
    "Stick",
    "compute_features",
    "sticks_to_vector",
    "score",
    "add_replicate",
    "build_index",
    "sim_traces",
]
