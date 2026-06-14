"""Experimental hybrid blocks for future SDF+GS stitching."""

from .sdf_densify_block import SDFDensifyBlock, SDFDensifyConfig
from .fm_sds_block import FlowMatchingSDSLikeBlock, FMGuidanceConfig
from .frequency_block import FrequencyDecompositionBlock, FrequencyLossConfig
from .scaffold_geometry_block import ScaffoldGeometryBlock, ScaffoldGeometryConfig
from .sof_prior_block import SOFPriorBlock, SOFPriorConfig
from .sof_regularization_block import SOFRegularizationBlock, SOFRegularizationConfig

__all__ = [
    "SDFDensifyBlock",
    "SDFDensifyConfig",
    "FlowMatchingSDSLikeBlock",
    "FMGuidanceConfig",
    "FrequencyDecompositionBlock",
    "FrequencyLossConfig",
    "ScaffoldGeometryBlock",
    "ScaffoldGeometryConfig",
    "SOFPriorBlock",
    "SOFPriorConfig",
    "SOFRegularizationBlock",
    "SOFRegularizationConfig",
]
