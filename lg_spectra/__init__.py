"""Little Garden HPLC/DAD spectral analysis toolkit."""

from .io import load_any, load_folder, save_spec
from .preprocess import apply_pipeline, PipelineStep, preset
from .sticks import pick_sticks, Stick
from .features import compute_features
from .vectorize import sticks_to_vector
from .matching import score
from .library import add_replicate, build_index
from .pipeline import run_pipelines, PIPELINE_PRESETS
from .sim import sim_traces

__all__ = [
    "load_any",
    "load_folder",
    "save_spec",
    "apply_pipeline",
    "PipelineStep",
    "preset",
    "pick_sticks",
    "Stick",
    "compute_features",
    "sticks_to_vector",
    "score",
    "add_replicate",
    "build_index",
    "run_pipelines",
    "PIPELINE_PRESETS",
    "sim_traces",
]
