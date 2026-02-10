from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class MethodConfig:
    name: str
    target_model: str
    draft_model: Optional[str] = None
    max_new_tokens: int = 128
    k: int = 4
    device: str = "cpu"
    dtype: str = "auto"
    quant: Optional[str] = None


@dataclass
class MethodResult:
    method: str
    metrics: Dict[str, float]
    outputs: Optional[List[str]] = None


class MethodRunner:
    """Interface for future decoding/judging methods."""

    def run(self, prompts: Iterable[str], config: MethodConfig) -> MethodResult:
        raise NotImplementedError
