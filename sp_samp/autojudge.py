from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .hf_adapter import HFModel


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _torch_multinomial(probs: torch.Tensor) -> int:
    token = torch.multinomial(probs, num_samples=1)
    return int(token.item())


def _safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def judge_features(
    probs: torch.Tensor,
    token_id: int,
    position_in_block: int,
    block_size: int,
) -> torch.Tensor:
    """Create lightweight draft-only features for judge classifier."""
    top_values, top_indices = torch.topk(probs, k=min(2, probs.numel()))
    max_prob = float(top_values[0])
    second_prob = float(top_values[1]) if top_values.numel() > 1 else 0.0
    token_prob = float(probs[token_id])
    entropy = float(-(probs * _safe_log(probs)).sum())
    is_top1 = 1.0 if int(top_indices[0]) == int(token_id) else 0.0
    pos_norm = float(position_in_block + 1) / float(max(block_size, 1))
    return torch.tensor(
        [
            token_prob,
            max_prob,
            second_prob,
            max_prob - second_prob,
            entropy,
            is_top1,
            pos_norm,
        ],
        dtype=torch.float32,
    )


class JudgeMLP(nn.Module):
    def __init__(self, in_features: int = 7, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AutoJudgeTrainConfig:
    max_train_samples: int = 4000
    max_new_tokens: int = 96
    k: int = 4
    train_steps: int = 400
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 123
    eps: float = 1e-12


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

    audit_samples: int = 0
    audit_expected_accept: float = 0.0

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

    @property
    def audit_expected_accept_mean(self) -> float:
        if self.audit_samples == 0:
            return 0.0
        return self.audit_expected_accept / self.audit_samples


def _stack_features(features: Sequence[torch.Tensor]) -> torch.Tensor:
    if not features:
        raise ValueError("No features collected for judge training.")
    return torch.stack(features, dim=0)


def collect_autojudge_training_data(
    target_model: HFModel,
    draft_model: HFModel,
    prompts: Iterable[Sequence[int]],
    cfg: AutoJudgeTrainConfig,
    eos_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect synthetic labels from exact acceptance probabilities."""
    torch.manual_seed(cfg.seed)
    features: List[torch.Tensor] = []
    labels: List[float] = []

    with torch.no_grad():
        for prompt_tokens in prompts:
            if len(features) >= cfg.max_train_samples:
                break
            generated: List[int] = []
            draft_state = draft_model.prefill(prompt_tokens)

            while len(generated) < cfg.max_new_tokens:
                remaining = cfg.max_new_tokens - len(generated)
                block = min(cfg.k, remaining)
                rejected = False

                for pos in range(block):
                    q_probs = _softmax(draft_state.logits).squeeze(0)
                    token = _torch_multinomial(q_probs)
                    feat = judge_features(q_probs, token, pos, cfg.k)

                    # Compute exact acceptance probability alpha via target.
                    prefix = list(prompt_tokens) + generated
                    target_state = target_model.prefill(prefix)
                    p_probs = _softmax(target_state.logits).squeeze(0)
                    q_token = float(q_probs[token])
                    p_token = float(p_probs[token])
                    alpha = min(1.0, p_token / max(q_token, cfg.eps))
                    accept_label = 1.0 if torch.rand(1).item() <= alpha else 0.0

                    features.append(feat.cpu())
                    labels.append(accept_label)
                    if len(features) >= cfg.max_train_samples:
                        break

                    if accept_label >= 0.5:
                        generated.append(token)
                        draft_state = draft_model.step([token], draft_state)
                        if eos_id is not None and token == eos_id:
                            rejected = True
                            break
                    else:
                        fallback = _torch_multinomial(p_probs)
                        generated.append(fallback)
                        draft_state = draft_model.prefill(list(prompt_tokens) + generated)
                        rejected = True
                        if eos_id is not None and fallback == eos_id:
                            break
                        break

                if len(features) >= cfg.max_train_samples or rejected:
                    if len(features) >= cfg.max_train_samples:
                        break
                if generated and eos_id is not None and generated[-1] == eos_id:
                    break

    x = _stack_features(features)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
    return x, y


def train_judge_classifier(
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 400,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 123,
    device: str = "cpu",
) -> Tuple[JudgeMLP, float]:
    torch.manual_seed(seed)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same number of rows.")
    if x.shape[0] == 0:
        raise ValueError("No training examples.")

    model = JudgeMLP(in_features=x.shape[1])
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]
    last_loss = 0.0

    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, size=(min(batch_size, n),), device=device)
        x_batch = x.index_select(0, idx)
        y_batch = y.index_select(0, idx)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    model.eval()
    return model, last_loss


def build_autojudge_classifier(
    target_model: HFModel,
    draft_model: HFModel,
    prompts: Iterable[Sequence[int]],
    cfg: AutoJudgeTrainConfig,
    eos_id: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[JudgeMLP, int, float]:
    x, y = collect_autojudge_training_data(
        target_model=target_model,
        draft_model=draft_model,
        prompts=prompts,
        cfg=cfg,
        eos_id=eos_id,
    )
    model, loss = train_judge_classifier(
        x=x,
        y=y,
        steps=cfg.train_steps,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
        device=device,
    )
    return model, int(x.shape[0]), float(loss)


def autojudge_sample_hf(
    target_model: HFModel,
    draft_model: HFModel,
    judge_model: JudgeMLP,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    k: int,
    threshold: float = 0.5,
    eos_id: Optional[int] = None,
    seed: Optional[int] = None,
    audit_ratio: float = 0.0,
    eps: float = 1e-12,
) -> Tuple[List[int], AutoJudgeStats]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if target_model.vocab_size != draft_model.vocab_size:
        raise ValueError("target and draft vocab sizes must match.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1].")
    if not (0.0 <= audit_ratio <= 1.0):
        raise ValueError("audit_ratio must be in [0, 1].")
    if seed is not None:
        torch.manual_seed(seed)

    generated: List[int] = []
    stats = AutoJudgeStats()

    judge_device = str(next(judge_model.parameters()).device)
    judge_model.eval()

    with torch.no_grad():
        draft_state = draft_model.prefill(prompt_tokens)
        stats.draft_prefills += 1

        while len(generated) < max_new_tokens:
            stats.steps += 1
            remaining = max_new_tokens - len(generated)
            block = min(k, remaining)
            rejected = False

            for pos in range(block):
                q_probs = _softmax(draft_state.logits).squeeze(0)
                token = _torch_multinomial(q_probs)

                feat = judge_features(q_probs, token, pos, k).to(judge_device)
                prob = float(torch.sigmoid(judge_model(feat.unsqueeze(0))).item())

                stats.proposed += 1
                stats.judge_total += 1

                if audit_ratio > 0.0 and torch.rand(1).item() < audit_ratio:
                    prefix = list(prompt_tokens) + generated
                    target_state = target_model.prefill(prefix)
                    stats.target_calls += 1
                    p_probs = _softmax(target_state.logits).squeeze(0)
                    alpha = min(1.0, float(p_probs[token]) / max(float(q_probs[token]), eps))
                    stats.audit_samples += 1
                    stats.audit_expected_accept += alpha

                if prob >= threshold:
                    stats.accepted += 1
                    stats.judge_accepted += 1
                    generated.append(token)
                    stats.target_tokens += 1
                    draft_state = draft_model.step([token], draft_state)
                    stats.draft_calls += 1
                    if eos_id is not None and token == eos_id:
                        return generated, stats
                    if len(generated) >= max_new_tokens:
                        return generated, stats
                else:
                    stats.rejections += 1
                    stats.judge_rejected += 1
                    stats.target_fallbacks += 1

                    prefix = list(prompt_tokens) + generated
                    target_state = target_model.prefill(prefix)
                    stats.target_calls += 1
                    p_probs = _softmax(target_state.logits).squeeze(0)
                    fallback = _torch_multinomial(p_probs)
                    generated.append(fallback)
                    stats.target_tokens += 1
                    if eos_id is not None and fallback == eos_id:
                        return generated, stats

                    draft_state = draft_model.prefill(list(prompt_tokens) + generated)
                    stats.draft_prefills += 1
                    rejected = True
                    break

            if rejected:
                continue

    return generated, stats
