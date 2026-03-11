from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple
import os
import tempfile
import warnings

import numpy as np
import torch

from .gsm8k import generations_equivalent
from .hf_adapter import HFModel

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import recall_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    LogisticRegression = None
    StandardScaler = None
    recall_score = None
    roc_auc_score = None


def _require_sklearn() -> None:
    if (
        LogisticRegression is None
        or StandardScaler is None
        or recall_score is None
        or roc_auc_score is None
    ):
        raise RuntimeError(
            "AutoJudge requires scikit-learn. Install project requirements first."
        )


def _default_c_grid() -> Tuple[float, ...]:
    # Paper: Section 3.2 — C-regularisation grid C ∈ {10^0, 10^-1, …, 10^-7}
    # (8 values; range(-7, 1) gives exponents -7,-6,-5,-4,-3,-2,-1,0)
    return tuple(10.0**p for p in range(-7, 1))


def parse_c_grid(spec: Optional[str]) -> Tuple[float, ...]:
    if spec is None or not spec.strip():
        return _default_c_grid()
    values: List[float] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(float(raw))
    if not values:
        return _default_c_grid()
    return tuple(values)


@dataclass
class AutoJudgeTrainConfig:
    task: str = "gsm8k"
    max_train_samples: int = 4000
    max_new_tokens: int = 96
    k: int = 4
    train_split: float = 0.9
    recall_target: float = 0.9
    c_grid: Tuple[float, ...] = _default_c_grid()
    seed: int = 123


@dataclass
class AutoJudgeStats:
    proposed: int = 0
    accepted: int = 0
    steps: int = 0
    target_tokens: int = 0
    rejections: int = 0

    judge_total: int = 0
    judge_accepted: int = 0
    judge_rejected: int = 0

    target_calls: int = 0
    target_fallbacks: int = 0
    draft_calls: int = 0
    draft_prefills: int = 0

    train_samples: int = 0
    train_loss: float = 0.0
    val_auc: float = 0.0
    val_recall: float = 0.0
    threshold_selected: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def avg_tokens_per_step(self) -> float:
        return self.target_tokens / self.steps if self.steps else 0.0

    @property
    def judge_accept_rate(self) -> float:
        return self.judge_accepted / self.judge_total if self.judge_total else 0.0

    @property
    def target_fallback_rate(self) -> float:
        return self.target_fallbacks / self.proposed if self.proposed else 0.0

    @property
    def target_calls_per_token(self) -> float:
        return self.target_calls / self.target_tokens if self.target_tokens else 0.0


@dataclass
class AutoJudgeClassifier:
    scaler: object
    model: object
    threshold: float
    feature_dim: int
    task: str = "gsm8k"
    c_selected: float = 0.0
    val_auc: float = 0.0
    val_recall: float = 0.0

    def predict_important_prob(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return self.model.predict_proba(x_scaled)[:, 1]


# Backward-compatible export name used by older code paths.
JudgeMLP = AutoJudgeClassifier


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


def _common_vocab_size(target_model: HFModel, draft_model: HFModel) -> int:
    sizes = (
        int(target_model.vocab_size),
        int(draft_model.vocab_size),
        _tokenizer_vocab_size(target_model),
        _tokenizer_vocab_size(draft_model),
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
    # Paper: Appendix A — paper uses Gumbel-max stochastic sampling; this implementation
    # uses argmax (greedy), a valid deterministic variant that does not affect correctness.
) -> List[int]:
    generated: List[int] = []
    if max_new_tokens <= 0:
        return generated

    with torch.no_grad():
        state = model.prefill(prompt_tokens)
        for _ in range(max_new_tokens):
            token = _argmax_token_from_logits(state.logits.squeeze(0))
            generated.append(token)
            if eos_id is not None and token == eos_id:
                break
            state = model.step([token], state)
    return generated


def _draft_argmax_tokens_for_target_response(
    draft_model: HFModel,
    prompt_tokens: Sequence[int],
    target_new_tokens: Sequence[int],
) -> List[int]:
    if not target_new_tokens:
        return []

    sequence = list(prompt_tokens) + list(target_new_tokens)
    logits, _ = draft_model.logits_and_last_hidden(sequence)
    start = len(prompt_tokens) - 1
    stop = start + len(target_new_tokens)
    if start < 0 or stop > logits.shape[1]:
        raise RuntimeError("Invalid logits slicing while mining important tokens.")
    return logits[0, start:stop, :].argmax(dim=-1).tolist()


def _feature_for_mismatch(
    target_model: HFModel,
    draft_model: HFModel,
    prompt_tokens: Sequence[int],
    target_prefix_tokens: Sequence[int],
    draft_token: int,
) -> torch.Tensor:
    prefix_with_draft = list(prompt_tokens) + list(target_prefix_tokens) + [int(draft_token)]
    _, draft_hidden = draft_model.logits_and_last_hidden(prefix_with_draft)
    _, target_hidden = target_model.logits_and_last_hidden(prefix_with_draft)
    draft_vec = draft_hidden[0, -1, :].detach().to(torch.float32).cpu()
    target_vec = target_hidden[0, -1, :].detach().to(torch.float32).cpu()
    return torch.cat([draft_vec, target_vec], dim=0)


def mine_important_tokens_gsm8k(
    target_model: HFModel,
    draft_model: HFModel,
    prompts: Iterable[Sequence[int]],
    cfg: AutoJudgeTrainConfig,
    eos_id: Optional[int] = None,
    mining_cache_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Paper: Algorithm 1 — important-token label mining for AutoJudge training.
    if cfg.task != "gsm8k":
        raise ValueError(f"Unsupported AutoJudge task: {cfg.task}")

    features: List[torch.Tensor] = []
    labels: List[float] = []
    prompts_done = 0

    # Resume from cache if available.
    if mining_cache_path is not None and os.path.exists(mining_cache_path):
        try:
            cache = torch.load(mining_cache_path, map_location="cpu", weights_only=False)
            saved_x = cache.get("x")
            saved_y = cache.get("y")
            prompts_done = int(cache.get("prompts_done", 0))
            if saved_x is not None and saved_y is not None and saved_x.shape[0] > 0:
                features = list(saved_x)
                labels = saved_y.squeeze(-1).tolist()
                print(
                    f"[{datetime.now():%H:%M:%S}] [mining] Resumed from cache:"
                    f" {prompts_done} prompts done, {len(features)} features loaded.",
                    flush=True,
                )
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] [mining] Cache load failed ({e}), starting fresh.", flush=True)
            features = []
            labels = []
            prompts_done = 0

    prompts_list = list(prompts)
    total_prompts = len(prompts_list)

    for prompt_tokens in prompts_list[prompts_done:]:
        if len(features) >= cfg.max_train_samples:
            break

        prompt = target_model.ensure_prefix(prompt_tokens)
        # Paper: Algorithm 1 line 1 — pseudocode says GENERATE(x, θ_draft); here we use
        # the TARGET model instead, which matches the paper's mathematical formula for I(x)
        # (importance defined relative to the target response). Intentional deviation from
        # the literal pseudocode; no correctness impact.
        y = _generate_greedy(
            model=target_model,
            prompt_tokens=prompt,
            max_new_tokens=cfg.max_new_tokens,
            eos_id=eos_id,
        )
        if not y:
            continue

        current = list(y)
        cursor = 0
        while cursor < len(current) and len(features) < cfg.max_train_samples:
            y_e = _draft_argmax_tokens_for_target_response(
                draft_model=draft_model,
                prompt_tokens=prompt,
                target_new_tokens=current,
            )
            mismatches = [i for i in range(cursor, len(current)) if current[i] != y_e[i]]
            if not mismatches:
                break

            t = mismatches[0]
            draft_token = int(y_e[t])

            feat = _feature_for_mismatch(
                target_model=target_model,
                draft_model=draft_model,
                prompt_tokens=prompt,
                target_prefix_tokens=current[:t],
                draft_token=draft_token,
            )

            replacement_prefix = prompt + current[:t] + [draft_token]
            alternative_suffix = _generate_greedy(
                model=target_model,
                prompt_tokens=replacement_prefix,
                max_new_tokens=max(cfg.max_new_tokens - (t + 1), 0),
                eos_id=eos_id,
            )
            y_hat = current[:t] + [draft_token] + alternative_suffix

            target_text = target_model.tokenizer.decode(current, skip_special_tokens=True)
            alt_text = target_model.tokenizer.decode(y_hat, skip_special_tokens=True)
            equivalent = generations_equivalent(target_text, alt_text)

            # Paper: Algorithm 1, line 9 — label convention: 0=unimportant, 1=important.
            important = 0.0 if equivalent else 1.0
            features.append(feat)
            labels.append(important)

            if equivalent:
                current = y_hat

            cursor = t + 1

        prompts_done += 1

        # Progress log + intermediate cache every 50 prompts.
        if prompts_done % 50 == 0:
            print(
                f"[{datetime.now():%H:%M:%S}] [mining]"
                f" {prompts_done}/{total_prompts} prompts, {len(features)} features",
                flush=True,
            )
            if mining_cache_path is not None and features:
                _save_mining_cache(mining_cache_path, features, labels, prompts_done)

    if not features:
        raise ValueError("No important-token samples were mined for AutoJudge training.")

    x = torch.stack(features, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    # Delete cache on successful completion.
    if mining_cache_path is not None and os.path.exists(mining_cache_path):
        try:
            os.remove(mining_cache_path)
        except OSError:
            pass

    return x, y


def _save_mining_cache(
    cache_path: str,
    features: List[torch.Tensor],
    labels: List[float],
    prompts_done: int,
) -> None:
    x = torch.stack(features, dim=0)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(cache_path) or ".", suffix=".tmp")
    try:
        os.close(tmp_fd)
        torch.save({"x": x, "y": y, "prompts_done": prompts_done}, tmp_path)
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


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


def _threshold_for_recall(
    probs: np.ndarray,
    labels: np.ndarray,
    recall_target: float,
) -> Tuple[float, float]:
    # Predict "important" if prob >= threshold.
    thresholds = np.unique(np.concatenate([probs, np.array([0.0, 1.0])]))
    thresholds = np.sort(thresholds)
    best_threshold = 0.5
    best_recall = 0.0

    for thr in thresholds:
        preds = (probs >= thr).astype(np.int64)
        rec = float(recall_score(labels, preds, zero_division=0))
        if rec >= recall_target:
            best_threshold = float(thr)
            best_recall = rec
    if best_recall == 0.0:
        preds = (probs >= 0.5).astype(np.int64)
        best_recall = float(recall_score(labels, preds, zero_division=0))
        best_threshold = 0.5
    return best_threshold, best_recall


def train_autojudge_logreg(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: AutoJudgeTrainConfig,
) -> Tuple[AutoJudgeClassifier, float]:
    # Paper: Section 3.2 — StandardScaler + LogisticRegression classifier with
    # C-grid cross-validation and recall-target threshold calibration on val split;
    # final model retrained on full dataset using best C.
    _require_sklearn()

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    if x.shape[0] < 2:
        raise ValueError("Need at least two AutoJudge training examples.")

    x_np = x.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy().reshape(-1).astype(np.int64)

    if len(np.unique(y_np)) < 2:
        raise ValueError("AutoJudge mined labels contain a single class; cannot train classifier.")

    train_idx, val_idx = _split_indices(x_np.shape[0], cfg.train_split, cfg.seed)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_np[train_idx])
    x_val = scaler.transform(x_np[val_idx])
    y_train = y_np[train_idx]
    y_val = y_np[val_idx]

    best_auc = -1.0
    best_model = None
    best_threshold = 0.5
    best_recall = 0.0
    best_c = float(cfg.c_grid[0])
    n_c = len(cfg.c_grid)

    for i, c in enumerate(cfg.c_grid):
        model = LogisticRegression(C=float(c), max_iter=500, random_state=cfg.seed)
        model.fit(x_train, y_train)
        val_probs = model.predict_proba(x_val)[:, 1]

        if len(np.unique(y_val)) < 2:
            val_auc = 0.0
        else:
            val_auc = float(roc_auc_score(y_val, val_probs))

        thr, rec = _threshold_for_recall(
            probs=val_probs,
            labels=y_val,
            recall_target=cfg.recall_target,
        )

        print(
            f"[{datetime.now():%H:%M:%S}] [autojudge-train]"
            f" C-grid {i + 1}/{n_c}: C={c:.2e} → val_AUC={val_auc:.4f}, recall={rec:.3f}",
            flush=True,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            best_threshold = thr
            best_recall = rec
            best_c = float(c)

    if best_model is None:
        raise RuntimeError("Failed to train AutoJudge classifier.")

    scaler_full = StandardScaler()
    x_full = scaler_full.fit_transform(x_np)
    model_full = LogisticRegression(C=best_c, max_iter=500, random_state=cfg.seed)
    model_full.fit(x_full, y_np)

    classifier = AutoJudgeClassifier(
        scaler=scaler_full,
        model=model_full,
        threshold=best_threshold,
        feature_dim=int(x_np.shape[1]),
        task=cfg.task,
        c_selected=best_c,
        val_auc=max(best_auc, 0.0),
        val_recall=max(best_recall, 0.0),
    )
    return classifier, classifier.val_auc


def train_judge_classifier(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Optional[AutoJudgeTrainConfig] = None,
) -> Tuple[AutoJudgeClassifier, float]:
    if cfg is None:
        cfg = AutoJudgeTrainConfig()
    return train_autojudge_logreg(x=x, y=y, cfg=cfg)


def build_autojudge_classifier(
    target_model: HFModel,
    draft_model: HFModel,
    prompts: Iterable[Sequence[int]],
    cfg: AutoJudgeTrainConfig,
    eos_id: Optional[int] = None,
    device: str = "cpu",
    mining_cache_path: Optional[str] = None,
) -> Tuple[AutoJudgeClassifier, int, float]:
    del device  # sklearn-based classifier is CPU-side.

    x, y = mine_important_tokens_gsm8k(
        target_model=target_model,
        draft_model=draft_model,
        prompts=prompts,
        cfg=cfg,
        eos_id=eos_id,
        mining_cache_path=mining_cache_path,
    )
    classifier, val_auc = train_autojudge_logreg(x=x, y=y, cfg=cfg)
    return classifier, int(x.shape[0]), float(val_auc)


def autojudge_sample_hf(
    target_model: HFModel,
    draft_model: HFModel,
    judge_model: AutoJudgeClassifier,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    threshold: Optional[float] = None,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[List[int], AutoJudgeStats]:
    del eps

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    common_vocab_n = _common_vocab_size(target_model, draft_model)

    if seed is not None:
        torch.manual_seed(seed)

    if threshold is None:
        threshold = float(judge_model.threshold)
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be in [0, 1].")

    generated: List[int] = []
    stats = AutoJudgeStats()
    base_prompt = target_model.ensure_prefix(prompt_tokens)

    while len(generated) < max_new_tokens:
        stats.steps += 1
        remaining = max_new_tokens - len(generated)
        block = min(k, remaining)

        prefix = list(base_prompt) + generated
        draft_state = draft_model.prefill(prefix)
        stats.draft_prefills += 1

        draft_tokens: List[int] = []
        for _ in range(block):
            token = _argmax_token_from_logits(draft_state.logits.squeeze(0)[:common_vocab_n])
            draft_tokens.append(token)
            stats.draft_calls += 1
            if eos_id is not None and token == eos_id:
                break
            draft_state = draft_model.step([token], draft_state)

        if not draft_tokens:
            break

        stats.proposed += len(draft_tokens)

        full_seq = prefix + draft_tokens
        target_logits, target_hidden = target_model.logits_and_last_hidden(full_seq)
        _, draft_hidden = draft_model.logits_and_last_hidden(full_seq)
        stats.target_calls += 1

        accepted_tokens: List[int] = []
        important_rejection = False
        start = len(prefix) - 1

        for i, draft_token in enumerate(draft_tokens):
            logits_row = target_logits[0, start + i, :common_vocab_n]
            target_token = int(torch.argmax(logits_row, dim=-1).item())

            if draft_token == target_token:
                accepted_tokens.append(draft_token)
                stats.accepted += 1
                continue

            # Paper: Section 3 — judge called only on mismatches for efficiency.
            stats.judge_total += 1
            abs_idx = len(prefix) + i
            feat = torch.cat(
                [
                    draft_hidden[0, abs_idx, :].to(torch.float32),
                    target_hidden[0, abs_idx, :].to(torch.float32),
                ],
                dim=0,
            )
            prob_important = float(
                judge_model.predict_important_prob(feat.cpu().numpy().reshape(1, -1))[0]
            )

            if prob_important < float(threshold):
                accepted_tokens.append(draft_token)
                stats.accepted += 1
                stats.judge_accepted += 1
                continue

            stats.judge_rejected += 1
            stats.rejections += 1
            stats.target_fallbacks += 1
            accepted_tokens.append(target_token)
            important_rejection = True
            break

        generated.extend(accepted_tokens)
        if len(generated) > max_new_tokens:
            generated = generated[:max_new_tokens]
        stats.target_tokens = len(generated)

        if generated and eos_id is not None and generated[-1] == eos_id:
            return generated, stats

        if len(generated) >= max_new_tokens:
            return generated, stats

        # Same as greedy speculative decoding: if full draft block accepted, emit one extra target token.
        if not important_rejection and len(accepted_tokens) == len(draft_tokens):
            extra_logits = target_logits[0, len(prefix) + len(draft_tokens) - 1, :]
            extra_token = int(torch.argmax(extra_logits, dim=-1).item())
            generated.append(extra_token)
            if len(generated) > max_new_tokens:
                generated = generated[:max_new_tokens]
            stats.target_tokens = len(generated)
            if eos_id is not None and generated[-1] == eos_id:
                return generated, stats

    return generated, stats


def warn_deprecated_autojudge_args() -> None:
    warnings.warn(
        "AutoJudge now uses paper-aligned LogisticRegression training. "
        "Legacy args (--autojudge-train-steps, --autojudge-train-batch-size, "
        "--autojudge-train-lr, --autojudge-audit-ratio) are ignored.",
        RuntimeWarning,
    )
