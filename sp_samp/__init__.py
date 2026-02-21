from .models import BaseModel, BigramModel, FixedModel, NoisyModel, RandomModel
from .mtbench import load_mtbench
from .sampling import SamplingStats, sample_baseline, speculative_sample
from .specexec import SpecExecStats, specexec_sample

__all__ = [
    "BaseModel",
    "BigramModel",
    "FixedModel",
    "NoisyModel",
    "RandomModel",
    "SamplingStats",
    "SpecExecStats",
    "load_mtbench",
    "sample_baseline",
    "speculative_sample",
    "specexec_sample",
]

# Optional torch/transformers-dependent exports.
try:
    from .hf_adapter import HFModel
    from .hf_sampling import sample_baseline_hf, speculative_sample_hf
    from .hf_specexec import specexec_sample_hf
    from .hf_topk import TopKStats, topk_sample_hf

    __all__ += [
        "HFModel",
        "sample_baseline_hf",
        "speculative_sample_hf",
        "specexec_sample_hf",
        "TopKStats",
        "topk_sample_hf",
    ]
except ModuleNotFoundError:
    pass

try:
    from .autojudge import (
        AutoJudgeClassifier,
        AutoJudgeStats,
        AutoJudgeTrainConfig,
        JudgeMLP,
        autojudge_sample_hf,
        build_autojudge_classifier,
        parse_c_grid,
    )

    __all__ += [
        "AutoJudgeClassifier",
        "AutoJudgeStats",
        "AutoJudgeTrainConfig",
        "JudgeMLP",
        "autojudge_sample_hf",
        "build_autojudge_classifier",
        "parse_c_grid",
    ]
except ModuleNotFoundError:
    pass
