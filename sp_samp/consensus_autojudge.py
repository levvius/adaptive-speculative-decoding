from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .hf_adapter import HFModel

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    LogisticRegression = None
    StandardScaler = None
    accuracy_score = None
    f1_score = None


ACTION_ACCEPT_D1 = 0
ACTION_ESCALATE_TO_D2 = 1
ACTION_FALLBACK_TO_TARGET = 2
ACTION_LABELS = (
    "accept_d1",
    "escalate_to_d2",
    "fallback_to_target",
)


def _require_sklearn() -> None:
    if (
        LogisticRegression is None
        or StandardScaler is None
        or accuracy_score is None
        or f1_score is None
    ):
        raise RuntimeError(
            "consensus_autojudge requires scikit-learn. Install project requirements first."
        )


def _normalize_feature_mode(spec: Optional[str]) -> str:
    mode = str(spec or "ensemble").strip().lower()
    if mode not in {"ensemble", "d1_only"}:
        raise ValueError(
            f"Unsupported consensus feature mode: {spec!r}. Expected one of: ensemble, d1_only."
        )
    return mode


def _normalize_gate_mode(spec: Optional[str]) -> str:
    mode = str(spec or "learned").strip().lower()
    if mode not in {"learned", "rule"}:
        raise ValueError(
            f"Unsupported consensus gate mode: {spec!r}. Expected one of: learned, rule."
        )
    return mode


def _argmax_token_from_logits(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def _tokenizer_vocab_size(model: HFModel) -> int:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return int(model.vocab_size)
    try:
        return int(len(tokenizer))
    except Exception:
        pass
    try:
        return int(len(tokenizer.get_vocab()))
    except Exception:
        return int(model.vocab_size)


def _common_vocab_size_three(
    target_model: HFModel,
    draft_model: HFModel,
    draft2_model: HFModel,
) -> int:
    sizes = (
        int(target_model.vocab_size),
        int(draft_model.vocab_size),
        int(draft2_model.vocab_size),
        _tokenizer_vocab_size(target_model),
        _tokenizer_vocab_size(draft_model),
        _tokenizer_vocab_size(draft2_model),
    )
    common = min(sizes)
    if common <= 0:
        raise ValueError(f"common vocab size must be positive, got {common} from {sizes}.")
    return common


def _generate_greedy(
    model: HFModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    eos_id: Optional[int] = None,
) -> List[int]:
    generated: List[int] = []
    if max_new_tokens <= 0:
        return generated

    with torch.no_grad():
        state = model.prefill(prompt_tokens)
        for step_idx in range(max_new_tokens):
            token = _argmax_token_from_logits(state.logits.squeeze(0))
            generated.append(token)
            if eos_id is not None and token == eos_id:
                break
            if step_idx + 1 >= max_new_tokens:
                break
            state = model.step([token], state)
    return generated


def _split_indices(n: int, train_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        return np.arange(n), np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_size = int(round(float(train_split) * n))
    train_size = max(1, min(train_size, n - 1))
    train_idx = idx[:train_size]
    val_idx = idx[train_size:]
    if val_idx.size == 0:
        val_idx = train_idx
    return train_idx, val_idx


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.to(torch.float32), dim=-1)


def _entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    p = probs.clamp(min=eps)
    return float((-(p * p.log()).sum()).item())


def _top1_margin(probs: torch.Tensor) -> float:
    if probs.numel() <= 1:
        return float(probs[0].item()) if probs.numel() == 1 else 0.0
    values = torch.topk(probs, k=min(2, int(probs.shape[0])), dim=-1).values
    if values.numel() == 1:
        return float(values[0].item())
    return float((values[0] - values[1]).item())


def _rank_of_token(probs: torch.Tensor, token_id: int) -> int:
    token_prob = probs[int(token_id)]
    higher = int((probs > token_prob).sum().item())
    return higher + 1


def _token_in_topk(probs: torch.Tensor, token_id: int, topk: int) -> bool:
    k_eff = min(max(int(topk), 1), int(probs.shape[0]))
    topk_ids = torch.topk(probs, k=k_eff, dim=-1).indices
    return bool((topk_ids == int(token_id)).any().item())


def _truncated_js_divergence(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    top_m: int,
    eps: float = 1e-12,
) -> float:
    m_eff = min(max(int(top_m), 1), int(probs_a.shape[0]), int(probs_b.shape[0]))
    ids_a = torch.topk(probs_a, k=m_eff, dim=-1).indices
    ids_b = torch.topk(probs_b, k=m_eff, dim=-1).indices
    merged_ids = torch.unique(torch.cat([ids_a, ids_b], dim=0), sorted=True)
    pa = probs_a.index_select(0, merged_ids)
    pb = probs_b.index_select(0, merged_ids)
    rest_a = torch.clamp(1.0 - pa.sum(), min=0.0)
    rest_b = torch.clamp(1.0 - pb.sum(), min=0.0)
    pa = torch.cat([pa, rest_a.view(1)], dim=0)
    pb = torch.cat([pb, rest_b.view(1)], dim=0)
    mean = 0.5 * (pa + pb)
    pa = pa.clamp(min=eps)
    pb = pb.clamp(min=eps)
    mean = mean.clamp(min=eps)
    kl_a = (pa * (pa.log() - mean.log())).sum()
    kl_b = (pb * (pb.log() - mean.log())).sum()
    return float((0.5 * (kl_a + kl_b)).item())


def _extract_ensemble_features(
    d1_logits_row: torch.Tensor,
    d2_logits_row: torch.Tensor,
    *,
    agreement_streak: int,
    block_position: int,
    block_size: int,
    top_m: int = 8,
    feature_mode: str = "ensemble",
) -> torch.Tensor:
    mode = _normalize_feature_mode(feature_mode)
    d1_probs = _softmax_probs(d1_logits_row.detach().cpu())
    d2_probs = _softmax_probs(d2_logits_row.detach().cpu())

    y1 = int(torch.argmax(d1_probs, dim=-1).item())
    y2 = int(torch.argmax(d2_probs, dim=-1).item())
    p1_y1 = float(d1_probs[y1].item())
    p2_y1 = float(d2_probs[y1].item())
    p1_y2 = float(d1_probs[y2].item())
    p2_y2 = float(d2_probs[y2].item())
    margin1 = _top1_margin(d1_probs)
    margin2 = _top1_margin(d2_probs)
    entropy1 = _entropy(d1_probs)
    entropy2 = _entropy(d2_probs)
    js_div = _truncated_js_divergence(d1_probs, d2_probs, top_m=top_m)
    rank_y1_d2 = float(min(_rank_of_token(d2_probs, y1), top_m + 1))
    rank_y2_d1 = float(min(_rank_of_token(d1_probs, y2), top_m + 1))
    in_topk_y1_d2 = 1.0 if _token_in_topk(d2_probs, y1, top_m) else 0.0
    in_topk_y2_d1 = 1.0 if _token_in_topk(d1_probs, y2, top_m) else 0.0
    position_ratio = float(block_position) / float(max(block_size, 1))
    streak_scaled = float(min(agreement_streak, top_m)) / float(max(top_m, 1))

    if mode == "d1_only":
        values = [
            p1_y1,
            margin1,
            entropy1,
            streak_scaled,
            position_ratio,
        ]
        return torch.tensor(values, dtype=torch.float32)

    values = [
        1.0 if y1 == y2 else 0.0,
        in_topk_y1_d2,
        in_topk_y2_d1,
        rank_y1_d2 / float(top_m + 1),
        rank_y2_d1 / float(top_m + 1),
        p1_y1,
        p2_y1,
        p1_y2,
        p2_y2,
        margin1,
        margin2,
        entropy1,
        entropy2,
        js_div,
        p2_y2 - p1_y1,
        streak_scaled,
        position_ratio,
    ]
    return torch.tensor(values, dtype=torch.float32)


def _draft_logits_on_target_path(
    model: HFModel,
    prompt_tokens: Sequence[int],
    target_new_tokens: Sequence[int],
    common_vocab_n: int,
) -> torch.Tensor:
    sequence = list(prompt_tokens) + list(target_new_tokens)
    logits, _ = model.logits_and_last_hidden(sequence)
    start = len(prompt_tokens) - 1
    stop = start + len(target_new_tokens)
    if start < 0 or stop > logits.shape[1]:
        raise RuntimeError("Invalid logits slicing while mining consensus labels.")
    return logits[0, start:stop, :common_vocab_n].detach().cpu()


@dataclass
class ConsensusAutoJudgeTrainConfig:
    task: str = "gsm8k"
    max_train_samples: int = 4000
    max_new_tokens: int = 96
    k: int = 4
    train_split: float = 0.9
    seed: int = 123
    top_m: int = 8
    feature_mode: str = "ensemble"
    fallback_threshold: float = 0.5


@dataclass
class ConsensusAutoJudgeStats:
    proposed: int = 0
    accepted: int = 0
    steps: int = 0
    target_tokens: int = 0
    rejections: int = 0

    gate_total: int = 0
    gate_accept_d1: int = 0
    gate_escalate_d2: int = 0
    gate_fallback: int = 0

    accepted_d1: int = 0
    accepted_d2: int = 0

    target_calls: int = 0
    target_fallbacks: int = 0
    draft_calls: int = 0
    draft_prefills: int = 0
    draft2_calls: int = 0
    draft2_prefills: int = 0

    train_samples: int = 0
    val_accuracy: float = 0.0
    val_macro_f1: float = 0.0
    fallback_threshold_selected: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def avg_tokens_per_step(self) -> float:
        return self.target_tokens / self.steps if self.steps else 0.0

    @property
    def target_fallback_rate(self) -> float:
        return self.target_fallbacks / self.proposed if self.proposed else 0.0

    @property
    def target_calls_per_token(self) -> float:
        return self.target_calls / self.target_tokens if self.target_tokens else 0.0

    @property
    def draft_calls_per_token(self) -> float:
        return self.draft_calls / self.target_tokens if self.target_tokens else 0.0

    @property
    def draft2_calls_per_token(self) -> float:
        return self.draft2_calls / self.target_tokens if self.target_tokens else 0.0

    @property
    def gate_accept_d1_rate(self) -> float:
        return self.gate_accept_d1 / self.gate_total if self.gate_total else 0.0

    @property
    def gate_escalate_rate(self) -> float:
        return self.gate_escalate_d2 / self.gate_total if self.gate_total else 0.0

    @property
    def gate_fallback_rate(self) -> float:
        return self.gate_fallback / self.gate_total if self.gate_total else 0.0


@dataclass
class ConsensusGateClassifier:
    scaler: object
    model: object
    feature_dim: int
    feature_mode: str = "ensemble"
    top_m: int = 8
    fallback_threshold: float = 0.5
    task: str = "gsm8k"
    val_accuracy: float = 0.0
    val_macro_f1: float = 0.0
    class_counts: Tuple[int, int, int] = (0, 0, 0)
    classes: Tuple[int, ...] = (0, 1, 2)

    def predict_action_probs(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_in = self.scaler.transform(x)
        probs = self.model.predict_proba(x_in)
        if probs.ndim != 2:
            raise RuntimeError(f"Expected 2D probabilities, got shape={probs.shape}.")
        expanded = np.zeros((probs.shape[0], 3), dtype=np.float64)
        for col, cls in enumerate(self.classes):
            expanded[:, int(cls)] = probs[:, col]
        return expanded

    def predict_action(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_action_probs(x)
        return np.argmax(probs, axis=1).astype(np.int64)


def mine_consensus_training_examples_gsm8k(
    target_model: HFModel,
    draft_model: HFModel,
    draft2_model: HFModel,
    prompts: Iterable[Sequence[int]],
    cfg: ConsensusAutoJudgeTrainConfig,
    eos_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cfg.task != "gsm8k":
        raise ValueError(f"Unsupported consensus_autojudge task: {cfg.task}")

    feature_mode = _normalize_feature_mode(cfg.feature_mode)
    common_vocab_n = _common_vocab_size_three(target_model, draft_model, draft2_model)
    features: List[torch.Tensor] = []
    labels: List[int] = []

    for prompt_tokens in prompts:
        if len(features) >= cfg.max_train_samples:
            break
        prompt = target_model.ensure_prefix(prompt_tokens)
        target_new = _generate_greedy(
            model=target_model,
            prompt_tokens=prompt,
            max_new_tokens=cfg.max_new_tokens,
            eos_id=eos_id,
        )
        if not target_new:
            continue

        d1_rows = _draft_logits_on_target_path(
            model=draft_model,
            prompt_tokens=prompt,
            target_new_tokens=target_new,
            common_vocab_n=common_vocab_n,
        )
        d2_rows = _draft_logits_on_target_path(
            model=draft2_model,
            prompt_tokens=prompt,
            target_new_tokens=target_new,
            common_vocab_n=common_vocab_n,
        )

        agreement_streak = 0
        for pos, target_token in enumerate(target_new):
            if len(features) >= cfg.max_train_samples:
                break
            d1_row = d1_rows[pos]
            d2_row = d2_rows[pos]
            d1_token = _argmax_token_from_logits(d1_row)
            d2_token = _argmax_token_from_logits(d2_row)
            feature = _extract_ensemble_features(
                d1_logits_row=d1_row,
                d2_logits_row=d2_row,
                agreement_streak=agreement_streak,
                block_position=pos % max(cfg.k, 1),
                block_size=max(cfg.k, 1),
                top_m=cfg.top_m,
                feature_mode=feature_mode,
            )
            if d1_token == int(target_token):
                label = ACTION_ACCEPT_D1
            elif d2_token == int(target_token):
                label = ACTION_ESCALATE_TO_D2
            else:
                label = ACTION_FALLBACK_TO_TARGET
            features.append(feature)
            labels.append(label)
            if d1_token == d2_token:
                agreement_streak += 1
            else:
                agreement_streak = 0

    if not features:
        raise ValueError("No consensus_autojudge training samples were mined.")

    x = torch.stack(features, dim=0)
    y = torch.tensor(labels, dtype=torch.int64)
    return x, y


def train_consensus_gate_classifier(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Optional[ConsensusAutoJudgeTrainConfig] = None,
) -> Tuple[ConsensusGateClassifier, float, float]:
    if cfg is None:
        cfg = ConsensusAutoJudgeTrainConfig()
    _require_sklearn()
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("x must be 2D and y must be 1D.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    if x.shape[0] < 3:
        raise ValueError("Need at least three consensus_autojudge training examples.")

    x_np = x.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().astype(np.int64)
    unique = np.unique(y_np)
    if unique.size < 2:
        raise ValueError("Consensus labels contain a single class; cannot train gate.")

    train_idx, val_idx = _split_indices(x_np.shape[0], cfg.train_split, cfg.seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_np[train_idx])
    x_val = scaler.transform(x_np[val_idx])
    y_train = y_np[train_idx]
    y_val = y_np[val_idx]

    model = LogisticRegression(
        max_iter=500,
        random_state=cfg.seed,
        solver="lbfgs",
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    val_preds = model.predict(x_val)
    val_accuracy = float(accuracy_score(y_val, val_preds))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

    scaler_full = StandardScaler()
    x_full = scaler_full.fit_transform(x_np)
    model_full = LogisticRegression(
        max_iter=500,
        random_state=cfg.seed,
        solver="lbfgs",
        class_weight="balanced",
    )
    model_full.fit(x_full, y_np)

    class_counts = tuple(int((y_np == cls).sum()) for cls in range(3))
    classifier = ConsensusGateClassifier(
        scaler=scaler_full,
        model=model_full,
        feature_dim=int(x_np.shape[1]),
        feature_mode=_normalize_feature_mode(cfg.feature_mode),
        top_m=int(cfg.top_m),
        fallback_threshold=float(cfg.fallback_threshold),
        task=cfg.task,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        class_counts=class_counts,
        classes=tuple(int(cls) for cls in model_full.classes_),
    )
    return classifier, val_accuracy, val_macro_f1


def build_consensus_gate_classifier(
    target_model: HFModel,
    draft_model: HFModel,
    draft2_model: HFModel,
    prompts: Iterable[Sequence[int]],
    cfg: ConsensusAutoJudgeTrainConfig,
    eos_id: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[ConsensusGateClassifier, int, float, float]:
    del device
    x, y = mine_consensus_training_examples_gsm8k(
        target_model=target_model,
        draft_model=draft_model,
        draft2_model=draft2_model,
        prompts=prompts,
        cfg=cfg,
        eos_id=eos_id,
    )
    classifier, val_accuracy, val_macro_f1 = train_consensus_gate_classifier(x=x, y=y, cfg=cfg)
    return classifier, int(x.shape[0]), val_accuracy, val_macro_f1


def _rule_gate_action(
    d1_logits_row: torch.Tensor,
    d2_logits_row: torch.Tensor,
    *,
    top_m: int,
    agreement_streak: int,
    disable_escalation: bool,
) -> int:
    del agreement_streak
    d1_probs = _softmax_probs(d1_logits_row.detach().cpu())
    d2_probs = _softmax_probs(d2_logits_row.detach().cpu())
    y1 = int(torch.argmax(d1_probs, dim=-1).item())
    y2 = int(torch.argmax(d2_probs, dim=-1).item())
    margin1 = _top1_margin(d1_probs)
    margin2 = _top1_margin(d2_probs)
    js_div = _truncated_js_divergence(d1_probs, d2_probs, top_m=top_m)

    if y1 == y2 and min(margin1, margin2) >= 0.10 and js_div <= 0.20:
        return ACTION_ACCEPT_D1

    if not disable_escalation:
        d2_top_prob = float(d2_probs[y2].item())
        d1_top_prob = float(d1_probs[y1].item())
        supports_d2 = _token_in_topk(d1_probs, y2, 2)
        if supports_d2 and d2_top_prob >= 0.35 and margin2 >= margin1 and js_div <= 0.45:
            return ACTION_ESCALATE_TO_D2
        if supports_d2 and d2_top_prob >= d1_top_prob:
            return ACTION_ESCALATE_TO_D2

    return ACTION_FALLBACK_TO_TARGET


def _select_action(
    d1_logits_row: torch.Tensor,
    d2_logits_row: torch.Tensor,
    *,
    gate_model: Optional[ConsensusGateClassifier],
    gate_mode: str,
    feature_mode: str,
    top_m: int,
    fallback_threshold: float,
    agreement_streak: int,
    block_position: int,
    block_size: int,
    disable_escalation: bool,
) -> int:
    mode = _normalize_gate_mode(gate_mode)
    if mode == "rule":
        return _rule_gate_action(
            d1_logits_row=d1_logits_row,
            d2_logits_row=d2_logits_row,
            top_m=top_m,
            agreement_streak=agreement_streak,
            disable_escalation=disable_escalation,
        )

    if gate_model is None:
        raise ValueError("gate_model is required when gate_mode='learned'.")
    feature = _extract_ensemble_features(
        d1_logits_row=d1_logits_row,
        d2_logits_row=d2_logits_row,
        agreement_streak=agreement_streak,
        block_position=block_position,
        block_size=block_size,
        top_m=top_m,
        feature_mode=feature_mode,
    )
    probs = gate_model.predict_action_probs(feature.numpy())[0]
    if probs[ACTION_FALLBACK_TO_TARGET] >= float(fallback_threshold):
        return ACTION_FALLBACK_TO_TARGET
    if disable_escalation:
        return ACTION_ACCEPT_D1
    non_fallback = probs[:2]
    return int(np.argmax(non_fallback))


def consensus_autojudge_sample_hf(
    target_model: HFModel,
    draft_model: HFModel,
    draft2_model: HFModel,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    gate_model: Optional[ConsensusGateClassifier] = None,
    gate_mode: str = "learned",
    fallback_threshold: Optional[float] = None,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    top_m: Optional[int] = None,
    feature_mode: Optional[str] = None,
    disable_escalation: bool = False,
) -> Tuple[List[int], ConsensusAutoJudgeStats]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    gate_mode = _normalize_gate_mode(gate_mode)
    if seed is not None:
        torch.manual_seed(seed)

    common_vocab_n = _common_vocab_size_three(target_model, draft_model, draft2_model)
    if gate_model is not None:
        inferred_top_m = int(gate_model.top_m)
        inferred_feature_mode = gate_model.feature_mode
        inferred_threshold = float(gate_model.fallback_threshold)
    else:
        inferred_top_m = 8
        inferred_feature_mode = "ensemble"
        inferred_threshold = 0.5
    top_m_eff = int(top_m) if top_m is not None else inferred_top_m
    feature_mode_eff = _normalize_feature_mode(
        feature_mode if feature_mode is not None else inferred_feature_mode
    )
    fallback_threshold_eff = (
        float(fallback_threshold) if fallback_threshold is not None else inferred_threshold
    )
    if not (0.0 <= fallback_threshold_eff <= 1.0):
        raise ValueError("fallback_threshold must be in [0, 1].")

    generated: List[int] = []
    stats = ConsensusAutoJudgeStats()
    base_prompt = target_model.ensure_prefix(prompt_tokens)

    with torch.no_grad():
        d1_state = draft_model.prefill(base_prompt)
        stats.draft_prefills += 1
        d2_state = draft2_model.prefill(base_prompt)
        stats.draft2_prefills += 1
        agreement_streak = 0

        while len(generated) < max_new_tokens:
            stats.steps += 1
            remaining = max_new_tokens - len(generated)
            block = min(k, remaining)
            fallback_triggered = False

            for block_pos in range(block):
                d1_row = d1_state.logits.squeeze(0)[:common_vocab_n]
                d2_row = d2_state.logits.squeeze(0)[:common_vocab_n]
                d1_token = _argmax_token_from_logits(d1_row)
                d2_token = _argmax_token_from_logits(d2_row)
                stats.draft_calls += 1
                stats.draft2_calls += 1
                stats.proposed += 1
                stats.gate_total += 1

                action = _select_action(
                    d1_logits_row=d1_row,
                    d2_logits_row=d2_row,
                    gate_model=gate_model,
                    gate_mode=gate_mode,
                    feature_mode=feature_mode_eff,
                    top_m=top_m_eff,
                    fallback_threshold=fallback_threshold_eff,
                    agreement_streak=agreement_streak,
                    block_position=block_pos,
                    block_size=block,
                    disable_escalation=disable_escalation,
                )

                if action == ACTION_FALLBACK_TO_TARGET:
                    prefix = list(base_prompt) + generated
                    target_logits, _ = target_model.logits_and_last_hidden(prefix)
                    target_token = _argmax_token_from_logits(target_logits[0, -1, :common_vocab_n])
                    generated.append(target_token)
                    stats.target_calls += 1
                    stats.target_fallbacks += 1
                    stats.gate_fallback += 1
                    stats.rejections += 1
                    stats.target_tokens = len(generated)
                    agreement_streak = 0
                    if eos_id is not None and target_token == eos_id:
                        return generated, stats
                    if len(generated) >= max_new_tokens:
                        return generated, stats
                    d1_state = draft_model.step([target_token], d1_state)
                    d2_state = draft2_model.step([target_token], d2_state)
                    fallback_triggered = True
                    break

                if action == ACTION_ESCALATE_TO_D2 and not disable_escalation:
                    emitted = d2_token
                    stats.gate_escalate_d2 += 1
                    stats.accepted_d2 += 1
                else:
                    emitted = d1_token
                    stats.gate_accept_d1 += 1
                    stats.accepted_d1 += 1

                generated.append(emitted)
                stats.accepted += 1
                stats.target_tokens = len(generated)
                if d1_token == d2_token:
                    agreement_streak += 1
                else:
                    agreement_streak = 0
                if eos_id is not None and emitted == eos_id:
                    return generated, stats
                if len(generated) >= max_new_tokens:
                    return generated, stats
                d1_state = draft_model.step([emitted], d1_state)
                d2_state = draft2_model.step([emitted], d2_state)

            if fallback_triggered:
                continue

    return generated, stats
