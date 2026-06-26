# Defense block-v1 bounded validation run

Date: 2026-06-27  
Planned start: 13:00 MSK  
Budget: at most 24 hours wall-clock, with the script budget set to 23 hours
(`TOTAL_BUDGET_SECONDS=82800`) so there is time left for inspection.

This run is a bounded validation of the repaired JointAdaSpec
`block_verify_v1` pipeline. It is **not final block-v1 benchmark evidence** and
must not replace the defense boundary that treats historical Run 1 as
`legacy / pre-repair evidence`.

## Command

```bash
tmux new -s blockv1_24h
cd /home/kubsu6/Projects/adaptive-speculative-decoding
git switch codex/defense-vkr-2026-06-26

make setup-gpu
make check

TOTAL_BUDGET_SECONDS=82800 \
DEFENSE_DATE=2026-06-27 \
MODEL_PAIR=qwen14b_0p5b \
bash scripts/run_defense_blockv1_24h.sh
```

Monitoring:

```bash
tail -f logs/defense_blockv1_2026-06-27.log
cat logs/defense_blockv1_2026-06-27.status.json
```

## Configuration

- Primary pair: `qwen14b_0p5b`
- Fallback pair: `qwen7b_1p5b`
- Fresh traces: `300`
- Benchmark prompts: `60`
- Seeds: `1`
- Max new tokens: `192`
- Active benchmark controls: `vanilla_ar`, `fixed_sd`,
  `cascade_verif_then_length`, plus `jointadaspec`

The expected primary output root is:

```text
outputs/jointadaspec_qwen14b_0p5b_defense_blockv1_2026-06-27/
```

If the fallback pair is used, replace `qwen14b_0p5b` with `qwen7b_1p5b` in the
output path.

## Current Local Preflight Status

On 2026-06-26, the current machine could not start the GPU validation because
`nvidia-smi` could not communicate with the NVIDIA driver. If this remains true
on 2026-06-27, do **not** run a large model job on this machine. Keep the
defense claim to CPU checks, CI, repaired loaders, and the reproducible rerun
protocol.

## Interpretation Rule

If the run succeeds, the only safe defense sentence is:

> A bounded fresh block-v1 validation run completed before the defense. It
> validates the repaired end-to-end pipeline, but it is not powered enough to
> serve as final statistical evidence.

If the run fails or is skipped, the safe defense sentence is:

> The 24-hour GPU validation was not available on this machine before the
> defense, so I do not present new block-v1 benchmark claims. The defended
> artifact remains the repaired implementation, semantic artifact validation,
> CPU/CI checks, and the reproducible rerun protocol.
