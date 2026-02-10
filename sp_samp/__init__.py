from .models import BaseModel, BigramModel, FixedModel, NoisyModel, RandomModel
from .mtbench import load_mtbench
from .sampling import SamplingStats, sample_baseline, speculative_sample

__all__ = [
    "BaseModel",
    "BigramModel",
    "FixedModel",
    "NoisyModel",
    "RandomModel",
    "SamplingStats",
    "load_mtbench",
    "sample_baseline",
    "speculative_sample",
]

# Optional torch/transformers-dependent exports.
try:
    from .hf_adapter import HFModel
    from .hf_sampling import sample_baseline_hf, speculative_sample_hf

    __all__ += [
        "HFModel",
        "sample_baseline_hf",
        "speculative_sample_hf",
    ]
except ModuleNotFoundError:
    pass

try:
    from .autojudge import (
        AutoJudgeStats,
        AutoJudgeTrainConfig,
        JudgeMLP,
        autojudge_sample_hf,
        build_autojudge_classifier,
    )

    __all__ += [
        "AutoJudgeStats",
        "AutoJudgeTrainConfig",
        "JudgeMLP",
        "autojudge_sample_hf",
        "build_autojudge_classifier",
    ]
except ModuleNotFoundError:
    pass
