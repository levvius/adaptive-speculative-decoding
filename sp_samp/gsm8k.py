from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class GSM8KSample:
    question: str
    answer: str


_FINAL_ANSWER_RE = re.compile(
    r"the\s+final\s+answer\s+is\s*[:\-]?\s*([^\n\r<]+)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _iter_json_items(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload
        return

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            for item in payload["data"]:
                if isinstance(item, dict):
                    yield item
            return
        yield payload


def load_gsm8k(path: str, max_samples: Optional[int] = None) -> List[GSM8KSample]:
    samples: List[GSM8KSample] = []
    for item in _iter_json_items(Path(path)):
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        samples.append(GSM8KSample(question=question, answer=answer))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def extract_reference_answer(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", maxsplit=1)[-1].strip()
    return answer.strip()


def extract_final_answer(text: str) -> str:
    matches = _FINAL_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    if "####" in text:
        return text.rsplit("####", maxsplit=1)[-1].strip()
    numbers = _NUMBER_RE.findall(text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return text.strip()


def parse_numeric_answer(text: str) -> Optional[Decimal]:
    value = text.strip().replace("$", "").replace(",", "")
    if not value:
        return None

    fraction = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", value)
    if fraction:
        num = Decimal(fraction.group(1))
        den = Decimal(fraction.group(2))
        if den == 0:
            return None
        return num / den

    numbers = _NUMBER_RE.findall(value)
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1])
    except (InvalidOperation, ValueError):
        return None


def answers_equivalent(lhs: str, rhs: str, tol: Decimal = Decimal("1e-6")) -> bool:
    left_num = parse_numeric_answer(lhs)
    right_num = parse_numeric_answer(rhs)
    if left_num is not None and right_num is not None:
        diff = abs(left_num - right_num)
        scale = max(abs(left_num), abs(right_num), Decimal("1"))
        return diff <= (tol * scale)

    left_norm = " ".join(lhs.strip().lower().split())
    right_norm = " ".join(rhs.strip().lower().split())
    return left_norm == right_norm


def generations_equivalent(lhs_text: str, rhs_text: str) -> bool:
    lhs_answer = extract_final_answer(lhs_text)
    rhs_answer = extract_final_answer(rhs_text)
    return answers_equivalent(lhs_answer, rhs_answer)
