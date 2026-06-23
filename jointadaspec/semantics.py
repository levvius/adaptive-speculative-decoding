"""Shared semantic-version metadata for repaired JointAdaSpec artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ACTION_SPACE_VERSION = 2
BENCHMARK_SCHEMA_VERSION = 3
TRACE_SCHEMA_VERSION = 3

BLOCK_DECODER_SEMANTICS = "block_verify_v1"
TARGET_ONLY_DECODER_SEMANTICS = "target_only_v1"
LEGACY_DECODER_SEMANTICS = "legacy_sequential"

TARGET_PASS_TARGET_ONLY = "target_only"
TARGET_PASS_BATCHED_BLOCK = "batched_block"
TARGET_PASS_SEQUENTIAL_FALLBACK = "sequential_fallback"
TARGET_PASS_LEGACY_SEQUENTIAL = "legacy_sequential"


def mdp_config_payload(config: Any) -> dict[str, Any]:
    """Return a stable JSON-serialisable representation of an MDP config."""
    if is_dataclass(config):
        data = asdict(config)
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        data = {
            key: getattr(config, key)
            for key in getattr(config, "__dataclass_fields__", {})
        }
    if "T_levels" in data:
        data["T_levels"] = [float(value) for value in data["T_levels"]]
    return data


def mdp_config_hash(config: Any) -> str:
    payload = mdp_config_payload(config)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def semantic_metadata(
    *,
    config: Any,
    policy_kind: str | None = None,
    num_actions: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "action_space_version": ACTION_SPACE_VERSION,
        "decoder_semantics": BLOCK_DECODER_SEMANTICS,
        "config": mdp_config_payload(config),
        "config_hash": mdp_config_hash(config),
    }
    if policy_kind is not None:
        metadata["policy_kind"] = str(policy_kind)
    if num_actions is not None:
        metadata["num_actions"] = int(num_actions)
    return metadata


def trace_metadata_path(traces_path: str | bytes | Path) -> Path:
    path = Path(traces_path)
    return path.with_name(f"{path.stem}_meta.json")


def require_semantic_metadata(
    metadata: Mapping[str, Any],
    *,
    artifact_label: str,
    expected_policy_kind: str | None = None,
    expected_num_actions: int | None = None,
    expected_config_hash: str | None = None,
) -> None:
    """Reject legacy artifacts that predate block-verification semantics."""
    missing = [
        key
        for key in ("action_space_version", "decoder_semantics", "config_hash")
        if key not in metadata
    ]
    if missing:
        raise ValueError(
            f"{artifact_label} is missing semantic metadata ({', '.join(missing)}). "
            "Regenerate it with action_space_version=2 and decoder_semantics=block_verify_v1."
        )

    try:
        version = int(metadata["action_space_version"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{artifact_label} has non-integer action_space_version="
            f"{metadata['action_space_version']!r}. Regenerate the artifact."
        ) from exc
    if version != ACTION_SPACE_VERSION:
        raise ValueError(
            f"{artifact_label} uses action_space_version={version}; expected "
            f"{ACTION_SPACE_VERSION}. Regenerate the artifact for the continue + verify@T action space."
        )

    semantics = str(metadata["decoder_semantics"])
    if semantics != BLOCK_DECODER_SEMANTICS:
        raise ValueError(
            f"{artifact_label} uses decoder_semantics={semantics!r}; expected "
            f"{BLOCK_DECODER_SEMANTICS!r}. Regenerate the artifact with block verification."
        )

    if expected_policy_kind is not None:
        policy_kind = str(metadata.get("policy_kind", ""))
        if policy_kind != expected_policy_kind:
            raise ValueError(
                f"{artifact_label} has policy_kind={policy_kind!r}; expected "
                f"{expected_policy_kind!r}."
            )

    if expected_num_actions is not None:
        try:
            num_actions = int(metadata["num_actions"])
        except KeyError as exc:
            raise ValueError(
                f"{artifact_label} is missing num_actions metadata. Regenerate the artifact."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{artifact_label} has non-integer num_actions={metadata.get('num_actions')!r}. "
                "Regenerate the artifact."
            ) from exc
        if num_actions != int(expected_num_actions):
            raise ValueError(
                f"{artifact_label} has num_actions={num_actions}; expected "
                f"{expected_num_actions}. This usually means a legacy action-space artifact was reused."
            )

    if expected_config_hash is not None and str(metadata["config_hash"]) != expected_config_hash:
        raise ValueError(
            f"{artifact_label} config_hash does not match the requested MDPConfig. "
            "Use traces/policies generated from the same config."
        )
