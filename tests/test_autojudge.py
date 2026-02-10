import torch

from sp_samp.autojudge import AutoJudgeStats, judge_features, train_judge_classifier


def test_judge_features_shape():
    probs = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32)
    feat = judge_features(probs, token_id=1, position_in_block=1, block_size=4)
    assert feat.shape == (7,)
    assert 0.0 <= float(feat[0]) <= 1.0
    assert 0.0 <= float(feat[6]) <= 1.0


def test_train_judge_classifier_learns_simple_boundary():
    torch.manual_seed(0)
    x = torch.rand(1200, 7)
    y = ((x[:, 0] + 0.8 * x[:, 3]) > 0.85).float().unsqueeze(-1)
    model, _ = train_judge_classifier(
        x=x,
        y=y,
        steps=250,
        batch_size=128,
        lr=5e-3,
        seed=0,
        device="cpu",
    )
    with torch.no_grad():
        logits = model(x)
        preds = (torch.sigmoid(logits) >= 0.5).float()
    accuracy = float((preds == y).float().mean().item())
    assert accuracy > 0.9


def test_autojudge_stats_rates():
    stats = AutoJudgeStats(
        proposed=10,
        accepted=7,
        steps=4,
        target_tokens=8,
        target_fallbacks=3,
        target_calls=2,
        judge_total=10,
        judge_accepted=7,
    )
    assert abs(stats.acceptance_rate - 0.7) < 1e-9
    assert abs(stats.avg_tokens_per_step - 2.0) < 1e-9
    assert abs(stats.judge_accept_rate - 0.7) < 1e-9
    assert abs(stats.target_fallback_rate - 0.3) < 1e-9
    assert abs(stats.target_calls_per_token - 0.25) < 1e-9
