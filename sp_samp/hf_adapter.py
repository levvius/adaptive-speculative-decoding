from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .models import BaseModel


def _load_local_config_json(model_name: str) -> Optional[Dict[str, Any]]:
    model_dir = Path(model_name)
    if not model_dir.is_dir():
        return None
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload
    return None


def _load_model_config_with_compat(model_name: str, trust_remote_code: bool):
    raw = _load_local_config_json(model_name)
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except KeyError:
        # Local Mistral-3 checkpoints may store text_config.model_type=ministral3.
        # Older transformers builds do not map this alias in AutoConfig.
        if raw is None:
            raise
        text_config = raw.get("text_config")
        if not isinstance(text_config, dict) or text_config.get("model_type") != "ministral3":
            raise
        patched = dict(raw)
        patched_text_config = dict(text_config)
        patched_text_config["model_type"] = "ministral"
        patched["text_config"] = patched_text_config
        try:
            from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
        except Exception:
            raise
        warnings.warn(
            "Applied local Mistral-3 config compatibility patch: "
            "text_config.model_type ministral3 -> ministral.",
            RuntimeWarning,
        )
        config = Mistral3Config(**patched)
    return config


def _ensure_mistral3_sliding_window(config) -> bool:
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return False
    text_model_type = str(getattr(text_config, "model_type", ""))
    if text_model_type not in {"ministral", "ministral3"}:
        return False
    if getattr(text_config, "sliding_window", None) is not None:
        return False
    text_config.sliding_window = 4096
    num_layers = int(getattr(text_config, "num_hidden_layers", 0) or 0)
    if num_layers > 0:
        text_config.layer_types = ["sliding_attention"] * num_layers
    return True


def _build_mistral3_fp8_dequant_config(config):
    if str(getattr(config, "model_type", "")) != "mistral3":
        return None
    quant_config = getattr(config, "quantization_config", None)
    quant_method = None
    if isinstance(quant_config, dict):
        quant_method = quant_config.get("quant_method")
    else:
        quant_method = getattr(quant_config, "quant_method", None)
    if str(quant_method).lower() != "fp8":
        return None
    try:
        from transformers import FineGrainedFP8Config
    except Exception:
        return None
    return FineGrainedFP8Config(dequantize=True)


def _load_tokenizer_with_fallback(
    model_name: str,
    tokenizer_name: Optional[str],
    use_fast_tokenizer: bool,
    trust_remote_code: bool,
):
    resolved_name = tokenizer_name or model_name
    try:
        return AutoTokenizer.from_pretrained(
            resolved_name,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
    except ValueError as exc:
        if "TokenizersBackend" not in str(exc):
            raise
        try:
            from transformers import MistralCommonBackend
        except Exception as import_exc:
            raise RuntimeError(
                "Tokenizer class TokenizersBackend requires mistral-common and a "
                "transformers build that exposes MistralCommonBackend."
            ) from import_exc
        warnings.warn(
            "Falling back to MistralCommonBackend tokenizer for local Mistral-3 checkpoint.",
            RuntimeWarning,
        )
        return MistralCommonBackend.from_pretrained(resolved_name)


@dataclass
class KVCacheState:
    past_key_values: tuple
    attention_mask: torch.Tensor
    logits: torch.Tensor
    length: int


class HFModel(BaseModel):
    """Hugging Face causal LM adapter with KV cache support."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        tokenizer_name: Optional[str] = None,
        use_fast_tokenizer: bool = True,
        quantization: Optional[str] = None,
        bnb_compute_dtype: str = "bfloat16",
    ) -> None:
        self.device = device
        self.tokenizer = _load_tokenizer_with_fallback(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            use_fast_tokenizer=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
        model_config = _load_model_config_with_compat(model_name, trust_remote_code)
        if _ensure_mistral3_sliding_window(model_config):
            warnings.warn(
                "Patched Mistral-3 text_config.sliding_window to 4096 for compatibility.",
                RuntimeWarning,
            )
        if dtype == "auto":
            torch_dtype = None
        else:
            if not hasattr(torch, dtype):
                raise ValueError(f"Unsupported dtype: {dtype}")
            torch_dtype = getattr(torch, dtype)

        quantization = quantization.lower() if quantization else None
        quantization_config = None
        device_map = None
        if quantization in {"8bit", "4bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError(
                    "Quantization requires bitsandbytes and a compatible transformers build."
                ) from exc
            if not hasattr(torch, bnb_compute_dtype):
                raise ValueError(f"Unsupported bnb_compute_dtype: {bnb_compute_dtype}")
            compute_dtype = getattr(torch, bnb_compute_dtype)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=quantization == "8bit",
                load_in_4bit=quantization == "4bit",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            device_map = "auto"
        mistral_fp8_dequant_config = None
        if quantization_config is None:
            mistral_fp8_dequant_config = _build_mistral3_fp8_dequant_config(model_config)
            if mistral_fp8_dequant_config is not None:
                warnings.warn(
                    "Detected FP8 Mistral-3 checkpoint; enabling dequantize=True for BF16-compatible runtime.",
                    RuntimeWarning,
                )

        def _load_model(q_config, d_map):
            kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
                "device_map": d_map,
                "config": model_config,
            }
            if q_config is not None:
                kwargs["quantization_config"] = q_config
            try:
                return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            except ValueError as exc:
                if (
                    str(getattr(model_config, "model_type", "")) == "mistral3"
                    and "Unrecognized configuration class" in str(exc)
                ):
                    from transformers.models.mistral3.modeling_mistral3 import (
                        Mistral3ForConditionalGeneration,
                    )

                    warnings.warn(
                        "Falling back to Mistral3ForConditionalGeneration for local Mistral-3 checkpoint.",
                        RuntimeWarning,
                    )
                    return Mistral3ForConditionalGeneration.from_pretrained(model_name, **kwargs)
                raise

        def _raise_cuda_arch_hint(exc: RuntimeError) -> None:
            msg = str(exc).lower()
            if "no kernel image is available for execution on the device" not in msg:
                raise exc
            arch = "unknown"
            supported_arches: List[str] = []
            if torch.cuda.is_available():
                try:
                    major, minor = torch.cuda.get_device_capability()
                    arch = f"sm_{major}{minor}"
                except Exception:
                    pass
                get_arch_list = getattr(torch.cuda, "get_arch_list", None)
                if get_arch_list is not None:
                    try:
                        supported_arches = [a for a in get_arch_list() if isinstance(a, str)]
                    except Exception:
                        supported_arches = []
            supported = ", ".join(supported_arches) if supported_arches else "unknown"
            raise RuntimeError(
                "Current torch build is incompatible with this GPU architecture "
                f"(device arch: {arch}, torch supports: {supported}). "
                "Use a torch build with native support for your GPU (for RTX 50xx: "
                "recommended torch>=2.7 with cu128+)."
            ) from exc

        load_quantization_config = (
            quantization_config if quantization_config is not None else mistral_fp8_dequant_config
        )

        try:
            self.model = _load_model(load_quantization_config, device_map)
        except ValueError as exc:
            msg = str(exc)
            if "does not recognize this architecture" in msg:
                raise RuntimeError(
                    "Transformers build is too old for this model architecture. "
                    "Upgrade transformers (for gpt_oss use >=4.55), rebuild image, "
                    "and retry."
                ) from exc
            incompatible_quant = (
                load_quantization_config is not None
                and "quantized with" in msg
                and "but you are passing a" in msg
            )
            if incompatible_quant:
                if quantization_config is not None:
                    warnings.warn(
                        "Model ships with its own quantization config; "
                        "ignoring --quant override and retrying with checkpoint defaults.",
                        RuntimeWarning,
                    )
                    quantization_config = None
                    device_map = "auto"
                    self.model = _load_model(mistral_fp8_dequant_config, device_map)
                elif mistral_fp8_dequant_config is not None:
                    warnings.warn(
                        "FineGrainedFP8 dequantize override was rejected by this checkpoint; "
                        "retrying with checkpoint quantization defaults.",
                        RuntimeWarning,
                    )
                    self.model = _load_model(None, device_map)
                else:
                    raise
            else:
                raise
        except RuntimeError as exc:
            _raise_cuda_arch_hint(exc)
        if device_map is None:
            try:
                self.model.to(device)
            except RuntimeError as exc:
                _raise_cuda_arch_hint(exc)
        self.device = str(next(self.model.parameters()).device)
        self.model.eval()
        vocab_size = getattr(self.model.config, "vocab_size", None)
        if vocab_size is None:
            text_config = getattr(self.model.config, "text_config", None)
            vocab_size = getattr(text_config, "vocab_size", None)
        if vocab_size is None:
            get_output_embeddings = getattr(self.model, "get_output_embeddings", None)
            if callable(get_output_embeddings):
                output_embeddings = get_output_embeddings()
                if output_embeddings is not None:
                    weight = getattr(output_embeddings, "weight", None)
                    if weight is not None and weight.ndim == 2:
                        vocab_size = int(weight.shape[0])
        if vocab_size is None:
            raise RuntimeError("Unable to resolve model vocab_size from config or output embeddings.")
        super().__init__(int(vocab_size))

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    def ensure_prefix(self, tokens: Sequence[int]) -> List[int]:
        if tokens:
            return list(tokens)
        if self.bos_token_id is not None:
            return [int(self.bos_token_id)]
        if self.eos_token_id is not None:
            return [int(self.eos_token_id)]
        return [0]

    def prefill(self, tokens: Sequence[int]) -> KVCacheState:
        input_ids = torch.tensor(
            [self.ensure_prefix(tokens)], device=self.device, dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        logits = outputs.logits[:, -1, :]
        return KVCacheState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            logits=logits,
            length=input_ids.shape[1],
        )

    def step(self, new_tokens: Sequence[int], state: KVCacheState) -> KVCacheState:
        if not new_tokens:
            return state
        input_ids = torch.tensor([list(new_tokens)], device=self.device, dtype=torch.long)
        new_mask = torch.ones_like(input_ids, device=self.device)
        attention_mask = torch.cat([state.attention_mask, new_mask], dim=1)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        return KVCacheState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            logits=logits,
            length=state.length + input_ids.shape[1],
        )

    def logits_and_last_hidden(self, tokens: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a full forward pass and return logits and final-layer hidden states."""
        input_ids = torch.tensor(
            [self.ensure_prefix(tokens)], device=self.device, dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        logits = outputs.logits
        if getattr(outputs, "hidden_states", None):
            hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            raise RuntimeError("Model forward output does not expose hidden states.")
        return logits, hidden

    def next_token_probs(self, context_tokens: Sequence[int]) -> List[float]:
        tokens = self.ensure_prefix(context_tokens)
        input_ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return self._validate(probs)
