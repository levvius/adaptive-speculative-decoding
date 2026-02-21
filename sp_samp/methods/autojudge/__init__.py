"""AutoJudge method exports."""

try:
    from ...autojudge import (
        AutoJudgeClassifier,
        AutoJudgeStats,
        AutoJudgeTrainConfig,
        JudgeMLP,
        autojudge_sample_hf,
        build_autojudge_classifier,
        parse_c_grid,
    )

    __all__ = [
        "AutoJudgeClassifier",
        "AutoJudgeStats",
        "AutoJudgeTrainConfig",
        "JudgeMLP",
        "autojudge_sample_hf",
        "build_autojudge_classifier",
        "parse_c_grid",
    ]
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    __all__ = []

    def __getattr__(name):
        raise ModuleNotFoundError(
            "AutoJudge dependencies are missing. Install torch, transformers, and scikit-learn first."
        ) from _IMPORT_ERROR
