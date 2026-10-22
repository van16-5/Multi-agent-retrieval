from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from .config import Config


class LLM:
    """Single shared generator used by every agent (Qwen3-4B-Instruct, 4-bit)."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.cfg = cfg
        self.calls = 0
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
        kwargs: Dict[str, Any] = {"device_map": "auto"}
        # T4 (compute capability 7.5) only emulates bfloat16, which is very slow; use float16 there.
        native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        if cfg.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(cfg.llm_model, **kwargs)
        self.model.eval()
        self._torch = torch

    def generate(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # enable_thinking=False only matters for hybrid-reasoning checkpoints (e.g. Qwen3-8B); others ignore it.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.cfg.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.cfg.temperature and self.cfg.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.cfg.temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class MockLLM:
    """Stand-in generator for testing the pipeline without a GPU.

    `responder(system, user) -> str` lets tests script agent replies.
    """

    def __init__(self, responder: Optional[Callable[[str, str], str]] = None):
        self.calls = 0
        self.responder = responder or (lambda s, u: "{}")

    def generate(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        self.calls += 1
        return self.responder(system, user)


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model reply, tolerating code fences and trailing text."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
                        try:
                            obj = json.loads(attempt)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                    break
        start = text.find("{", start + 1)
    return None


def as_list(x) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]
