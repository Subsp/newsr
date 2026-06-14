"""Geometry prior loaders for hybrid SDF+GS training."""

from .scaffold_provider import ScaffoldData, ScaffoldLoadConfig, load_scaffold_data

__all__ = [
    "ScaffoldData",
    "ScaffoldLoadConfig",
    "load_scaffold_data",
]
