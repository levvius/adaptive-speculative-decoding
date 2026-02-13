"""AutoJudge method exports."""

try:
    from ...autojudge import (
        AutoJudgeStats,
        AutoJudgeTrainConfig,
        JudgeMLP,
        autojudge_sample_hf,
        build_autojudge_classifier,
    )

    __all__ = [
        "AutoJudgeStats",
        "AutoJudgeTrainConfig",
        "JudgeMLP",
        "autojudge_sample_hf",
        "build_autojudge_classifier",
    ]
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
    __all__ = []

    def __getattr__(name):
        raise ModuleNotFoundError(
            "AutoJudge dependencies are missing. Install torch and transformers first."
        ) from _IMPORT_ERROR
