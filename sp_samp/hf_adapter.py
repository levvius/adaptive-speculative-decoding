from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import BaseModel


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
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

        def _load_model(q_config, d_map):
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                quantization_config=q_config,
                device_map=d_map,
            )

        try:
            self.model = _load_model(quantization_config, device_map)
        except ValueError as exc:
            msg = str(exc)
            if "does not recognize this architecture" in msg:
                raise RuntimeError(
                    "Transformers build is too old for this model architecture. "
                    "Upgrade transformers (for gpt_oss use >=4.55), rebuild image, "
                    "and retry."
                ) from exc
            incompatible_quant = (
                quantization_config is not None
                and "quantized with" in msg
                and "but you are passing a" in msg
            )
            if incompatible_quant:
                warnings.warn(
                    "Model ships with its own quantization config; "
                    "ignoring --quant override and retrying with checkpoint defaults.",
                    RuntimeWarning,
                )
                quantization_config = None
                device_map = "auto"
                self.model = _load_model(quantization_config, device_map)
            else:
                raise
        if device_map is None:
            self.model.to(device)
        self.device = str(next(self.model.parameters()).device)
        self.model.eval()
        super().__init__(int(self.model.config.vocab_size))

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

    def next_token_probs(self, context_tokens: Sequence[int]) -> List[float]:
        tokens = self.ensure_prefix(context_tokens)
        input_ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return self._validate(probs)
